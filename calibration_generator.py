import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import re

# am Ende dran denken den ORdner zu ändern! #
DATA_FOLDER = './EMA2025'

CHANNEL_HEIGHT = 15 # Kanalhöhe = 15 mm
       
        
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

    # 1. Grades Fit: slope ist dp/dx
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
                    # Lade alle Spalten (16) als 2D-Array
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
    rotor_frequencies: list
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
                records.append({
                    'sensor': sensor,
                    'direction': direction,
                    'rotor_frequency': freq,
                    'U_mean': Umean
                })
    return pd.DataFrame(records)

def fit_calibration_curves(df_all, degree=7):
    """
    Fittet pro Sensor in df_all ein Kalibrier-Polynom 
    tau_w = f(U) vom Grad `degree`.
    
    Parameters
    ----------
    df_all : pd.DataFrame
        Muss die Spalten ['sensor', 'U_mean', 'wandschubspannung'] enthalten.
    degree : int
        Grad des anzupassenden Polynoms (Standard 7).
    
    Returns
    -------
    dict
        sensor -> np.poly1d, die Kalibrierfunktion τ_w(U)
    """
    cal_curves = {}
    for sensor, group in df_all.groupby('sensor'):
        # Nutze jeweils nur die Daten dieses Sensors
        tau = group['tau_neg'].values.astype(float)
        U   = group['U_mean'].values.astype(float)
        coeffs = np.polyfit(tau, U, degree)
        cal_curves[sensor] = np.poly1d(coeffs)
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
        grp = df_all[df_all.sensor==s]
        tau = grp['tau_neg'].values.astype(float)
        U   = grp['U_mean'].values.astype(float)
        poly = cal_curves[s]

        tau_fit = np.linspace(tau.min(), tau.max(), 200)
        U_fit   = poly(tau_fit)

        plt.figure(figsize=(7,5))
        plt.scatter(tau, U, label='Messwerte')
        plt.plot(tau_fit, U_fit, '-', lw=2, label=f'{7}. Grades-Fit')
        plt.xlabel('τ_w [Pa]')
        plt.ylabel('U [V]')
        plt.title(f'Inverse Kalibrierkurve Sensor {s}')
        plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()
    
def save_cal_coefs(cal_curves):
    rows = []
    for s, poly in cal_curves.items():
        coefs = poly.coefficients[::-1]  # a0...a7
        rows.append({'sensor': s, **{f'a{i}': c for i,c in enumerate(coefs)}})
    pd.DataFrame(rows).to_csv('calibration_coefficients.csv', index=False)

def calibration_offset(df_all, degree=7, output_file='rms_offsets.txt'):
    """
    Fittet pro Sensor ein Kalibrier-Polynom tau = f(U) und berechnet den RMS-Offset.
    Speichert die Ergebnisse zusätzlich in einer TXT-Datei.

    Returns:
    --------
    dict: sensor -> (poly1d, RMS)
    """
    cal_curves = {}
    lines = []

    for sensor, group in df_all.groupby('sensor'):
        tau = group['tau_neg'].values.astype(float)
        U   = group['U_mean'].values.astype(float)
        # Polynomial fit: tau = f(U)
        coeffs = np.polyfit(tau, U, degree)
        poly = np.poly1d(coeffs)
        # Berechnung der Fitwerte und RMS
        U_fit = poly(tau)
        rms_offset = np.sqrt(np.mean((U - U_fit)**2))
        cal_curves[sensor] = (poly, rms_offset)
        print(f"Sensor: {sensor} | RMS-Offset: {rms_offset:.5f} V")
        lines.append(f"Sensor: {sensor} | RMS-Offset: {rms_offset:.5f} V\n")

    with open(output_file, 'w') as f:
        f.writelines(lines)

    return cal_curves
    
# ---------------------
#         MAIN
# ---------------------
if __name__ == "__main__":

    # 1) Messinfo einlesen
    sensors, directions, freqs, file_types = get_measurement_info_from_filenames(DATA_FOLDER)

    # 2) p- und U-Daten verarbeiten
    df_press = compute_pressure_means(DATA_FOLDER, sensors, directions, freqs)
    df_u     = compute_voltage_means(DATA_FOLDER, sensors, directions, freqs)

    # 3) Zusammenführen
    df_all = pd.merge(df_press, df_u, on=['sensor','direction','rotor_frequency'], how='inner')

    # 4) Sensor in Position 2 -> tau_w künstlich negativ
    df_all['tau_neg'] = df_all.apply(
        lambda r:  r['wandschubspannung'] if r['direction']=='1' else -r['wandschubspannung'],
        axis=1
        )

    # 5) Kalibrierkurven fitten
    cal_curves = fit_calibration_curves(df_all, degree=7)

    # 6) Beispielplot Druckgradient zur Veranschaulichung
    plot_example_pressure(df_all, sensor='ANW65', direction='1', rotor_frequency='45.00')
    
    # 7) Plots der Kalibrierungskurven
    plot_calcurve(df_all, cal_curves, sensors)
    
    # 8) Koeffizienten speichern
    save_cal_coefs(cal_curves)

    # 9) RMS-Offset ausgeben und in txt Datei speichern
    calibration_offset(df_all, degree=7)


