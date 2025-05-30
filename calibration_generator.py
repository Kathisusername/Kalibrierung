import re
import os
import numpy as np
from numpy.polynomial import polynomial as poly
import pandas as pd
import matplotlib.pyplot as plt

# am Ende dran denken den ORdner zu ändern! #
DATA_FOLDER = './EMA2025'

CHANNEL_HEIGHT = 15 # Kanalhöhe = 15 mm
DEGREE = 7
       
        
def get_measurement_info_from_filenames(data_folder):
    """
    Extrahiert einzigartige Sensornamen aus Dateinamen in einem angegebenen Ordner.
    Die Dateinamen folgen dem Muster 'ANWXX-D_RR.RR_Y.dat',
    wobei 'ANWXX' der Sensorname ist.

    Args:
        data_folder (str): Der Pfad zum Ordner, der die Messdaten enthält.

    Returns:
        list: Eine sortierte Liste von einzigartigen Sensornamen (z.B. ['ANW19', 'ANW22']).
              Gibt eine leere Liste zurück, wenn der Ordner nicht existiert oder keine passenden Dateien enthält.
    """
    all_sensor_names = set()
    all_directions = set()
    all_rotor_frequencies = set()
    all_file_types = set()

    if not os.path.exists(data_folder):
        print(f"Fehler: Der Ordner '{data_folder}' existiert nicht.")
        return []

    filename_pattern = re.compile(r'^(ANW\d+)-(\d)_(\d{2}\.\d{2})_([pU])\.dat$')
    
    for filename in os.listdir(data_folder):
        match = filename_pattern.match(filename)
        if match:
            sensor_name = match.group(1)
            direction = match.group(2)
            rotor_frequency = match.group(3)
            file_type = match.group(4) # p oder U

            all_sensor_names.add(sensor_name)
            all_directions.add(direction)
            all_rotor_frequencies.add(rotor_frequency)
            all_file_types.add(file_type)
    
    return all_sensor_names, all_directions, all_rotor_frequencies, all_file_types

def calculate_druckgradient(col_means):
    """
    Führt einen linearen Fit durch die Mittelwerte der 16 Druckspalten
    col_means (Liste [p1, p2, …, p16]) gegen die Positionen 0,100,…,1400,1600 mm
    durch und gibt nur die Steigung (dp/dx) in Pa/mm zurück.
    """
    y = np.asarray(col_means, float)
    n = len(y)

    # Erzeuge Positionen: erst 0,100,...,1400 und dann 1400+200=1600
    x = np.arange(n) * 100.0
    x[-1] = x[-2] + 200.0

    # 1. Grades Fit: slope ist Druckgradient dp/dx
    slope, _ = np.polyfit(x, y, 1)
    return slope

def calculate_wandschubspannung(dpdx, H=CHANNEL_HEIGHT):
    return -dpdx * (H / 2.0)

def compute_pressure_means(
    data_folder: str,
    sensor_names: list,
    directions: list,
    rotor_frequencies: list
) -> pd.DataFrame:
    """
    Für jede Kombination aus Sensor, Richtung und Frequenz lädt die _p.dat-Datei,
    berechnet den Mittelwert jeder der 16 Druck-Spalten und packt die Ergebnisse in ein DataFrame.
    Spalten im DataFrame:
      - sensor
      - direction
      - rotor_frequency
      - p1, p2, …, p16  (jeweils Mittelwert der entsprechenden Spalte)
    """
    records = []
    for sensor in sensor_names:
        for direction in directions:
            for freq in rotor_frequencies:
                fname = f"{sensor}-{direction}_{freq}_p.dat"
                fpath = os.path.join(data_folder, fname)
                if os.path.isfile(fpath):
                    # Lade alle Spalten (16) als Array
                    data = np.loadtxt(fpath)
                    # Berechne Mittelwert jeder Spalte
                    col_means = data.mean(axis=0)
                    # Druckgradient berechnen
                    dpdx = calculate_druckgradient(col_means)
                    # Wandschubspannung berechnen
                    tauw = calculate_wandschubspannung(dpdx, CHANNEL_HEIGHT)
                    # Record aufbauen
                    record = {
                        'sensor': sensor,
                        'direction': direction,
                        'rotor_frequency': freq,
                        'druckgradient': dpdx, 
                        'wandschubspannung': tauw
                    }
                    # Füge p1…p16 hinzu
                    for idx, m in enumerate(col_means, start=1):
                        record[f'p{idx}'] = m
                    records.append(record)
    # Erzeuge DataFrame mit allen Records
    df = pd.DataFrame(records)
    return df

def compute_voltage_means(
    data_folder: str,
    sensor_names: list,
    directions: list,
    rotor_frequencies: list,
    offsets_dict: dict = None  # Optional: manuelle Offsets
) -> pd.DataFrame:
    
    records = []
    for sensor in sensor_names:
        for direction in directions:
            for freq in rotor_frequencies:
                fname = f"{sensor}-{direction}_{freq}_U.dat"
                fpath = os.path.join(data_folder, fname)
                if not os.path.isfile(fpath):
                    continue
                data = np.loadtxt(fpath)
                Umean = data[:, 3].mean()

                # Offset abziehen, falls vorhanden
                if offsets_dict and sensor in offsets_dict:
                    Umean -= offsets_dict[sensor]
                       
                records.append({
                    'sensor': sensor,
                    'direction': direction,
                    'rotor_frequency': freq,
                    'U_mean': Umean
                })
    return pd.DataFrame(records)

def fit_calibration_curves(df_all, DEGREE):
    """
    Fittet pro Sensor in df_all ein Kalibrier-Polynom 
    tau_w = f(U) vom Grad `DEGREE` und berechnet MSE & Offset.

    Returns
    -------
    dict
        sensor -> {
            'poly':   Polynomial-Objekt P mit P(U)=tau_w,
            'mse':    mittlerer quadratischer Fehler zwischen tau_neg und P(U),
            'offset': mittlerer Fehler (tau_neg − P(U))
        }
    """
    cal_curves = {}
    for sensor, group in df_all.groupby('sensor'):
        U   = group['U_mean'].to_numpy(dtype=float)
        tau = group['tau_neg'].to_numpy(dtype=float)

        # 1) Fit Koeffizienten in „polynomial“-Basis
        coeffs = poly.polyfit(U, tau, DEGREE)
        # 2) daraus ein Polynomial-Objekt bauen
        P = poly.Polynomial(coeffs)

        tau_fit = P(U)
        mse = np.mean((tau-tau_fit)**2)
        offset = np.mean(tau-tau_fit)
        
        cal_curves[sensor] = {
            'poly': P,
            'mse': mse,
            'offset': offset
        }
        print(f"Sensor {sensor}: MSE={mse:.3e}, Offset={offset:.3e}")
        # 3) in unser Dict übernehmen
        cal_curves[sensor] = P

    return cal_curves

def plot_example_pressure(df, sensor, direction, rotor_frequency):
    example = df_all[(df_all.sensor=='ANW94') & (df_all.direction=='1') & (df_all.rotor_frequency=='45.00')]
    pos = np.arange(16) * 100.0
    pos[-1] = pos[-2] + 200.0
    ps = example[[f'p{i}' for i in range(1,17)]].iloc[0].values.astype(float)
    slope, intercept = np.polyfit(pos, ps, 1)
    fit_line = slope*pos + intercept

    plt.figure(figsize=(8,5))
    plt.plot(pos, ps, 'o-', label='Messwerte')
    plt.plot(pos, fit_line, '--', label=f'$dp/dx$={slope:.3f} Pa/mm')
    plt.xlabel('Position [mm]')
    plt.ylabel('Mittlerer Druck [Pa]')
    plt.title('Druckgradient ANW94, Richtung 1, 45 Hz')
    plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()
    
def plot_calcurve(df_all, cal_curves, sensors):
    for s in sensors:
        grp = df_all[df_all.sensor == s]
        U   = grp['U_mean'].values.astype(float)
        tau = grp['tau_neg'].values.astype(float)
        P   = cal_curves[s]

        U_fit   = np.linspace(U.min(), U.max(), 200)
        tau_fit = P(U_fit)

        plt.figure(figsize=(7,5))
        plt.scatter(tau, U, label='Messwerte', zorder=5)
        plt.plot(tau_fit, U_fit, '-', lw=2, label=f'Fit {DEGREE}. Grades')
        plt.xlabel('Wandschubspannung $\\tau_w$ [Pa]')
        plt.ylabel('Sensorspannung U[V]')
        plt.title(f'Kalibrierkurve Sensor {s}')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()   

def save_cal_coefs(cal_curves, output_file='calibration_coefficients.csv'):
    rows = []
    for sensor, P in cal_curves.items():
        # P.coef ist ein Array [c0, c1, ..., c_DEGREE]
        coeffs = P.coef  
        row = {'sensor': sensor}
        for i, ci in enumerate(coeffs):
            row[f'a{i}'] = ci
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(output_file, index=False)
    print(f"Calibration coefficients saved to {output_file}")

def plot_mse_vs_degree(
    df_all,
    tau_col='tau_neg',
    U_col='U_mean',
    degree_range=None,
    exclude_sensors=None,
    smooth_window: int = 5
):
    """
    Berechnet für jeden Sensor (außer den in `exclude_sensors`) und jeden Grad in `degree_range` den MSE
    zwischen den Messwerten (tau_col, U_col) und dem Polynomfit.
    Plottet anschließend MSE vs. Polynomgrad für alle übrigen Sensoren.

    Parameters
    ----------
    df_all : pd.DataFrame
        Muss Spalten ['sensor', tau_col, U_col] enthalten.
    tau_col : str
        Name der Spalte mit tau_w.
    U_col : str
        Name der Spalte mit Sensorspannung U.
    degree_range : iterable of int, optional
        Liste oder Range der getesteten Polynomegrade (Standard 1–10).
    exclude_sensors : iterable of str, optional
        Sensoren, die nicht geplottet werden sollen (Standard ['ANW89','ANW94']). <- defekte Sensoren
    """
    if degree_range is None:
        degree_range = range(1, 11)
    if exclude_sensors is None:
        exclude_sensors = ['ANW89', 'ANW94']

    # alle Sensoren, minus die Ausgeschlossenen
    sensors = [s for s in sorted(df_all['sensor'].unique())
               if s not in exclude_sensors]

    # MSE für jeden Sensor und jedes Polynomgrad sammeln
    mse_dict = {s: [] for s in sensors}
    for s in sensors:
        grp = df_all[df_all['sensor'] == s]
        tau = grp[tau_col].values.astype(float)
        U   = grp[U_col].values.astype(float)

        for deg in degree_range:
            coeffs = np.polyfit(tau, U, deg)
            U_pred = np.polyval(coeffs, tau)
            mse = np.mean((U - U_pred) ** 2)
            mse_dict[s].append(mse)

    # Plot
    plt.figure(figsize=(10, 6))
    degs = np.array(list(degree_range))
    for s, mses in mse_dict.items():
        # glätten
        mses_smooth = pd.Series(mses).rolling(
            window=smooth_window,
            center=True,
            min_periods=1
        ).mean().values
        plt.plot(degs, mses_smooth, marker='o', label=s)

    plt.xlabel('Polynomgrad')
    plt.ylabel('MSE [V²]')
    plt.title('MSE pro Polynomgrad (ohne ANW89 & ANW94)')
    plt.xticks(list(degree_range))
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    plt.tight_layout()
    plt.show()
    
  
# ---------------------
#         MAIN
# ---------------------
if __name__ == "__main__":

    # Offset definieren
    manual_offsets = {
        'ANW61': -0.0786,   # Im Laufe der Kalibrierung eingestellte Offsets
        'ANW43': 0.0005,
        'ANW49': 0.001,
        'ANW22': 0.005,
        'ANW110': 0.0002,
        'ANW88': 0,
        'ANW65': 0.035,
        'ANW19': 0,
        'ANW117': 0.005,
        'ANW109': -0.005,
        'ANW63': 0.061,
        'ANW94': 0.001,
    }
    
    # 1) Messinfo einlesen
    sensors, directions, freqs, file_types = get_measurement_info_from_filenames(DATA_FOLDER)

    # 2) p- und U-Daten verarbeiten
    df_press = compute_pressure_means(DATA_FOLDER, sensors, directions, freqs)
    #df_u     = compute_voltage_means(DATA_FOLDER, sensors, directions, freqs)
    df_u = compute_voltage_means(DATA_FOLDER, sensors, directions, freqs, offsets_dict=manual_offsets)

    # 3) Zusammenführen
    df_all = pd.merge(df_press, df_u, on=['sensor','direction','rotor_frequency'], how='inner')

    # 4) Sensor in Position 2 -> tau_w künstlich negativ
    df_all['tau_neg'] = df_all.apply(
        lambda r:  r['wandschubspannung'] if r['direction']=='1' else -r['wandschubspannung'],
        axis=1
        )

    # 5) Kalibrierkurven fitten
    cal_curves = fit_calibration_curves(df_all, DEGREE)


    # 6) Beispielplot Druckgradient zur Veranschaulichung
    plot_example_pressure(df_all, sensor='ANW65', direction='1', rotor_frequency='45.00')
    
    # 7) Plots der Kalibrierungskurven
    sensors = sorted(df_all['sensor'].unique())
    plot_calcurve(df_all, cal_curves, sensors)

    
    # 8) Koeffizienten speichern
    save_cal_coefs(cal_curves)


    #10) MSE über Polynomgrad plotten (Abweichung des Polynomfits von den Daten)
    plot_mse_vs_degree(df_all, smooth_window=5)