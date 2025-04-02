import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from lmfit import Model
from scipy.odr import ODR, Model, Data, RealData
from scipy.stats import norm
from scipy.optimize import curve_fit

r = np.array([0.15, 0.2, 0.15875, 0.238125, 0.3175, 0.396875, 0.47625, 0.555625, 0.635, 0.7144375])
r /= 2
err_r = 0.3

rho_s = 7.870
err_rho_s = 0.001

rho_l = 1.26
err_rho_l = 0.01

g = 9.806


def get_data(folder_path):
    """ Legge  i dati dal file .txt e crea il frame di Pandas"""
    dataframes = {}

    # Scansiona tutti i file nella cartella
    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)

        # Controlla se è un file e se ha un'estensione valida
        if os.path.isfile(file_path) and (file.endswith(".txt") or file.endswith(".csv")):
            try:
                # Legge il file con Pandas adattando il separatore
                df = pd.read_csv(file_path, sep='\t', header=None, encoding='utf-8')

                # Salva il DataFrame con il nome del file come chiave
                dataframes[file] = df

            except Exception as e:
                print(f"Errore nella lettura di {file}: {e}")

    return dataframes


def horizontal_model(B, x):
    """ horizontal model"""
    return B[0]


def horizontal_fit(x, y, err_x, err_y):
    """ Fit ortogonale con incertezze su x e y """

    model = Model(horizontal_model)
    data = RealData(x, y, sx=err_x, sy=err_y)  # sx e sy sono gli errori su x e y
    odr = ODR(data, model, beta0=[0])  # Stima iniziale [pendenza, intercetta]
    output = odr.run()
    alpha = output.beta
    err_alpha = np.sqrt(np.diag(output.cov_beta))  # Incertezze parametri

    return alpha, err_alpha, output


def exponential_model(x, a, b):
    """modello esponenziale"""
    return a * np.exp(-x / b)


def exponential_fit(x, y, err_y):
    """ Fit ortogonale con incertezze su x e y """
    popt, pcov = curve_fit(exponential_model, x, y, sigma=err_y, absolute_sigma=True)
    alpha, beta = popt
    err_alpha, err_beta = np.sqrt(np.diag(pcov))

    return alpha, beta, err_alpha, err_beta


def linear_model(B, x):
    """ linear model"""
    return B[0] * x + B[1]


def linear_fit(x, y, err_x, err_y):
    """ Fit ortogonale con incertezze su x e y """

    model = Model(linear_model)
    data = RealData(x, y, sx=err_x, sy=err_y)  # sx e sy sono gli errori su x e y
    odr = ODR(data, model, beta0=[0, 0])  # Stima iniziale [pendenza, intercetta]
    output = odr.run()
    alpha, beta = output.beta
    err_alpha, err_beta = np.sqrt(np.diag(output.cov_beta))  # Incertezze parametri

    return alpha, beta, err_alpha, err_beta, output


def quadratic_model(B, x):
    """ quadratic_model """
    return B[0] * x ** 2 + B[1] * x + B[2]


def quadratic_fit(x, y, err_x, err_y):
    """ Fit ortogonale con incertezze su x e y """

    model = Model(quadratic_model)
    data = RealData(x, y, sx=err_x, sy=err_y)  # sx e sy sono gli errori su x e y
    odr = ODR(data, model, beta0=[0, 0, 0])  # Stima iniziale [pendenza, intercetta]
    output = odr.run()
    alpha, beta, gamma = output.beta
    err_alpha, err_beta, err_gamma = np.sqrt(np.diag(output.cov_beta))  # Incertezze parametri

    return alpha, beta, gamma, err_alpha, err_beta, err_gamma, output


def cubic_model(B, x):
    """ cubic_model"""
    return B[0] * x ** 3 + B[1] * x ** 2 + B[2] * x + B[3]


def cubic_fit(x, y, err_x, err_y):
    """ Fit ortogonale con incertezze su x e y """

    model = Model(quadratic_model)
    data = RealData(x, y, sx=err_x, sy=err_y)  # sx e sy sono gli errori su x e y
    odr = ODR(data, model, beta0=[-0.01, -0.1, -0.01, 0])  # Stima iniziale [pendenza, intercetta]
    output = odr.run()
    alpha, beta, gamma, epsilon = output.beta
    err_alpha, err_beta, err_gamma, err_epsilon = np.sqrt(np.diag(output.cov_beta))  # Incertezze parametri

    return alpha, beta, gamma, epsilon, err_alpha, err_beta, err_gamma, err_epsilon, output


def manipulate_data(dataframe):
    """
    permette di fare un fit lineare di ciascun set di dati. Calcola il chi-quadro ridotto che sara' utilizzato
    come peso per calcolare la media del tempo per ciascun traguardo
    """
    Data = {}
    for name, df in dataframe.items():
        space = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
        dx = 5
        err_space = [0.2 / np.sqrt(12)]
        semi_cum_space = [0.5 * space[0]]  # prende il punto intermedio tra un valore e il successivo di spazio

        time = [df.iloc[0].mean()]
        err_time = [df.iloc[0].std() / np.sqrt(len(df.iloc[0]))]
        semi_cum_time = [0.5 * df.iloc[0].mean()]
        semi_cum_time_err = [0.5 * df.iloc[0].std() / np.sqrt(len(df.iloc[0]))]

        speed = [space[0] / time[0]]  # si associa al semi-spazio
        err_speed = [speed[0] * np.sqrt((err_space[0] / dx) ** 2 + (err_time[0] / time[0]) ** 2)]

        # l'incertezza sullo spazio e' la somma delle incertezze degli estremi dell'intervallo
        semi_cum_space_err = [2 * err_space[0]]

        # popola tutte le colonne utili
        for i in range(1, df.shape[0]):
            err_space.append(err_space[i - 1])

            semi_cum_space.append(space[i - 1] + 0.5 * space[0])
            semi_cum_space_err.append(2 * err_space[0])  # dipende solo dalla lettura delle tacche

            time.append(df.iloc[i].mean())
            err_time.append(df.iloc[i].std() / np.sqrt(len(df.iloc[i])))
            semi_cum_time.append(semi_cum_time[i - 1] + 0.5 * time[i - 1] + 0.5 * time[i])
            semi_cum_time_err.append(semi_cum_time_err[i - 1] + 0.5 * err_time[i])

            speed.append(dx / df.iloc[i].mean())
            err_speed.append(speed[i] * np.sqrt((err_space[i] / dx) ** 2 + (err_time[i] / time[i]) ** 2))

        #  converti gli array in Series per creare il DataFrame
        a0 = pd.Series(space, name="space")
        a1 = pd.Series(time, name="time")
        a2 = pd.Series(semi_cum_space, name="1/2_space")
        a3 = pd.Series(semi_cum_time, name="1/2_time")
        a4 = pd.Series(speed, name="speed")

        b0 = pd.Series(err_space, name="err_space")
        b1 = pd.Series(err_time, name="err_time")
        b2 = pd.Series(semi_cum_space_err, name="err_1/2_space")
        b3 = pd.Series(semi_cum_time_err, name="err_1/2_time")
        b4 = pd.Series(err_speed, name="err_speed")

        # print(name, a0)
        Data[name] = pd.concat([a0, a1, a2, a3, a4, b0, b1, b2, b3, b4], axis=1)

    # print(Data)
    return Data


def manipulate_results(dataframe, Data):
    """
    Il Dataframe 'time' ha tante colonne quanti i diametri analizzati, sulle righe il tempo medio di caduta nelle
    diverse analisi
    """
    for nome, df in dataframe.items():
        for i in range(0, df.shape[1]):  # df.shape[1] sono le colonne
            for j in range(0, df.shape[0]):  # df.shape[0] sono le righe
                if not pd.isna(pd.to_numeric(df.at[j, i], errors='coerce')):
                    pass
                else:
                    # riempio il buco con la media dei valori delle altre misurazioni dello stesso intervallo di spazio
                    # aggiustata con un fattore dato dalla distribuzione gaussiana delle stesse
                    df.at[j, i] = Data[nome].at[j, "time"] + np.random.normal(scale=df.iloc[j].std())

    time = {}
    for nome, df in dataframe.items():
        time[nome] = dataframe[nome].sum()

    time = pd.DataFrame(time)

    return dataframe, time


def calcolemus(v_lim, err_v_lim, Data, choice):
    """
    Applicabile solo per fluido a velocita' limite
    Calcola la velocita' limite, da qui la viscosita
    """
    if choice:
        eta = []
        err_eta = []
        i = 0

        for nome, set in Data.items():
            eta_i= []
            err_eta_i = []

            space_12 = set["1/2_space"].to_numpy()
            time_12 = set["1/2_time"].to_numpy()
            speed = set["speed"].to_numpy()

            err_space_12 = set["err_1/2_space"].to_numpy()
            err_time_12 = set["err_1/2_time"].to_numpy()
            err_speed = set["err_speed"].to_numpy()

            mask = (
                    ~np.isnan(space_12) &
                    ~np.isnan(time_12) &
                    ~np.isnan(speed) &
                    ~np.isnan(err_space_12) &
                    ~np.isnan(err_time_12) &
                    ~np.isnan(err_speed)
            )

            space_12 = space_12[mask]
            time_12 = time_12[mask]
            speed = speed[mask]

            err_space_12 = err_space_12[mask]
            err_time_12 = err_time_12[mask]
            err_speed = err_speed[mask]

            alpha, beta, err_alpha, err_beta, output = linear_fit(time_12, space_12, err_x=err_time_12,
                                                                  err_y=err_space_12)

            for i in range(0, len(speed)):
                eta_i.append((2 * g * r[i] ** 2 * (rho_s - rho_l)) / (9 * speed[i]))

            # print(eta_i)

            eta.append((2 * g * r[i] ** 2 * (rho_s - rho_l)) / (9 * alpha))

            d_eta_dr = (4 * g * r[i] * (rho_s - rho_l)) / (9 * alpha)
            d_eta_drho_s = (2 * g * r[i] ** 2) / (9 * alpha)
            d_eta_drho_l = - (2 * g * r[i] ** 2) / (9 * alpha)
            d_eta_dalpha = - (2 * g * r[i] ** 2 * (rho_s - rho_l)) / (9 * alpha ** 2)

            # Propagazione dell'incertezza
            sigma_eta = np.sqrt(
                (d_eta_dr * err_r) ** 2 +
                (d_eta_drho_s * err_rho_s) ** 2 +
                (d_eta_drho_l * err_rho_l) ** 2 +
                (d_eta_dalpha * err_alpha) ** 2
            )

            err_eta.append(sigma_eta)
            i += 1
            #print(alpha, err_alpha)

        weights = 1 / np.array(err_eta) ** 2  # Pesi inversamente proporzionali al quadrato delle incertezze
        mean = np.sum(eta * weights) / np.sum(weights)
        sigma_mean = np.sqrt(1 / np.sum(weights))

        mean /= 10
        sigma_mean /= 10
        # divido per 10 per unita' di misura utilizzate

        plt.figure(figsize=(8, 6))
        plt.errorbar(r, eta, xerr=err_r, yerr=err_eta, fmt='o', capsize=5, capthick=1,
                     label=f"Media pesata: {mean / 10:.1f} Pa s\n"
                           f"Incertezza sulla media: {sigma_mean / 10:.1f} Pa s")
        plt.xlabel("raggio [mm]")
        plt.ylabel("eta [Pa s]")

        plt.title("Viscosita'", pad=20)
        plt.legend()
        plt.grid()
        # plt.show()

        return eta


def correction(Data):
    """
    Questo codice fa una corezione delle velocita' limite in funzione della
    colonna di fluido sotto la sfera = 59.5 - set["1/2_space"]
    e del raggio del cilindro = 4.65
    """
    prev_speed = {}
    i = 0
    for name, set in Data.items():
        set["speed"] *= (1 + 2.3 * r[i] / 4.65) * (1 + 3.3 * r[i] / (59.5 - set["1/2_space"]))
        i += 1

    return Data


def plot(Data, Time, dataframe, choice, head_skip, tail_skip):
    """
    Grafica i dati e restituire i valori utili all'analisi
    """
    dx = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    v_lim = []
    err_v_lim = []

    if choice < 10:
        for nome, set in Data.items():
            space = set["space"].to_numpy()
            time = set["time"].to_numpy()
            space_12 = set["1/2_space"].to_numpy()
            time_12 = set["1/2_time"].to_numpy()
            speed = set["speed"].to_numpy()

            err_space = set["err_space"].to_numpy()
            err_time = set["err_time"].to_numpy()
            err_space_12 = set["err_1/2_space"].to_numpy()
            err_time_12 = set["err_1/2_time"].to_numpy()
            err_speed = set["err_speed"].to_numpy()

            mask = (
                    ~np.isnan(space) &
                    ~np.isnan(time) &
                    ~np.isnan(space_12) &
                    ~np.isnan(time_12) &
                    ~np.isnan(speed) &
                    ~np.isnan(err_space) &
                    ~np.isnan(err_time) &
                    ~np.isnan(err_space_12) &
                    ~np.isnan(err_time_12) &
                    ~np.isnan(err_speed)
            )

            space = space[mask]
            time = time[mask]
            space_12 = space_12[mask]
            time_12 = time_12[mask]
            speed = speed[mask]

            err_space = err_space[mask]
            err_time = err_time[mask]
            err_space_12 = err_space_12[mask]
            err_time_12 = err_time_12[mask]
            err_speed = err_speed[mask]

            if choice == 1:
                x = time_12[:len(time_12)]
                y = speed[:len(speed)]
                err_x = err_time_12[:len(err_time_12):]
                err_y = err_speed[:len(err_speed)]

                fig, ax1 = plt.subplots(figsize=(10, 8))
                ax1.errorbar(x, y, xerr=err_x, yerr=err_y, fmt='o', capsize=5, capthick=1)

                ax1.set_xlabel("tempo [s]", fontsize=23, labelpad=15)
                ax1.set_xlim(0, x[-1] + x[0])
                ax1.set_ylabel("velocita' [cm/s]", fontsize=23, labelpad=15)

                ax2 = ax1.secondary_xaxis('top')
                ax2.set_xlabel("spazio [cm]", fontsize=23, labelpad=15)
                ax2.set_xticks(x)
                ax2.set_xticklabels(space_12, fontsize=18)

            if choice == 2:
                x = time_12
                y = space_12
                err_x = err_time_12
                err_y = err_space_12

                plt.figure(figsize=(8, 7))
                plt.errorbar(x, y, xerr=err_x, yerr=err_y, fmt='o', capsize=5, capthick=1)
                plt.xlabel("tempo [s]", fontsize=20, labelpad=15)
                plt.ylabel("spazio [cm]", fontsize=20,labelpad=15)

            if choice == 3:
                x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
                y = time
                err_x = 0.0001
                err_y = err_time

                plt.figure(figsize=(8, 6))
                plt.errorbar(x, y, xerr=err_x, yerr=err_y, fmt='o', capsize=5, capthick=1)
                plt.xlabel("indice")
                plt.ylabel("tempo [s]")

            if choice == 4:
                # print(dataframe)
                x = []
                for name, set in dataframe.items():
                    for i in len(set):
                        x.append(1)
                x = np.array(x)

            if choice == 5:
                x = time_12[head_skip:-tail_skip]
                y = speed[head_skip:-tail_skip]
                err_x = err_time_12[head_skip:-tail_skip]
                err_y = err_speed[head_skip:-tail_skip]

                fig, ax1 = plt.subplots(figsize=(13, 9))
                ax1.errorbar(x, y, xerr=err_x, yerr=err_y, fmt='o', capsize=5, capthick=1)

                ax1.set_xlabel("tempo [s]", fontsize=16)
                ax1.set_xlim(0, x[-1] + x[0])
                ax1.set_ylabel("velocita' [cm/s]", fontsize=16)

                ax2 = ax1.secondary_xaxis('top')
                ax2.set_xlabel("spazio [cm]", labelpad=15, fontsize=16)
                ax2.set_xticks(x, fontsize=20)
                ax2.set_xticklabels(space_12[head_skip:-tail_skip])

            alpha, beta, err_alpha, err_beta, output = linear_fit(x, y, err_x=err_x, err_y=err_y)
            y_fit = alpha * x + beta
            v_lim = alpha
            err_v_lim = err_alpha

            #y_fit = y.mean()
            #arr_y = np.repeat(y_fit, len(y))

            # print(alpha, err_alpha)

            chi2 = np.sum(((y - y_fit) / err_y) ** 2)
            dof = len(y) - 2
            chi2_red = chi2 / dof

            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)
            plt.plot(x, y_fit, color='orange', linewidth=1,
                     label=r"Fit $y = Ax + B$"
                           f"\nA: ({alpha:.4f} ± {err_alpha:.4f}) cm/s\n"
                           f"B: ({beta:.2f} ± {err_beta:.2f}) cm \n"
                           f"Chi2_rid: {chi2_red:.2f}")

            #plt.title(f"{nome}", pad=20)
            plt.legend(fontsize=16)
            plt.grid()
            plt.savefig(f"C:/Users/loren/Desktop/tempo/{nome[:len(nome) - 4]}.png")
            #plt.show()

        return v_lim, err_v_lim


    if choice == 20:
        """
            Permette di fare il grafico della PDF Tempo in funzione del diametro della sfera
        """

        labels = ["1.5mm", "2mm", "2_32", "3_32", "4_32", "5_32", "6_32", "7_32", "8_32", "9_32"]
        mean = Time.mean()
        std = Time.std()
        x_range = np.linspace(mean - 4 * std, mean + 4 * std, 200)
        pdf = norm.pdf(x_range, loc=mean, scale=std)
        plt.figure(figsize=(8, 6))
        plt.plot(x_range, pdf, label=labels)

        plt.title("PDF Tempo in funzione del diametro - Fluido non-Newtoniano", pad=20)
        plt.xlabel("Tempo [s]")
        plt.ylabel("PD")
        plt.legend()
        plt.grid(True)
        plt.show()

    if choice == 21:
        labels = ["1.5mm", "2mm", "2_32", "3_32", "4_32", "5_32", "6_32", "7_32", "8_32", "9_32"]
        # print(dataframe)
        plt.figure(figsize=(10, 6))

        for name, set in dataframe.items():
            # print(set)
            plt.hist(set.values.ravel(), bins=8, alpha=0.5, density=False, edgecolor='black', label=name)
        '''for name, set in Data.items():
            print(set["time"])
            #bins = np.linspace(Time.min().min(), Time.max().max(), 50)
            plt.hist(set["time"], bins=5, alpha=0.5, density=False, edgecolor='black', label=name)
'''
        plt.xlim([0, 60])
        plt.title("Fluido non-Newtoniano")
        plt.xlabel("Tempo [s]")
        plt.ylabel("Conteggi")
        plt.legend()
        plt.show()

    if choice == 22:
        for name, set in Data.items():
            x = set["1/2_time"].to_numpy()
            y = set["speed"].to_numpy()
            err_x = set["err_1/2_time"].to_numpy()
            err_y = set["err_speed"].to_numpy()

            print(x, y, err_x, err_y, "\n")

            alpha, beta, err_alpha, err_beta = exponential_fit(x, y, err_y=err_y)
            y_fit = alpha * np.exp(-x / beta)

            chi2 = np.sum(((y - y_fit) / err_y) ** 2)
            dof = len(y) - 2
            chi2_red = chi2 / dof

            plt.figure(figsize=(8, 6))
            plt.errorbar(x, y, xerr=err_x, yerr=err_y, fmt='o', capsize=5, capthick=1)
            plt.xlabel("tempo [s]")
            plt.ylabel("velocita' [cm/s]")

            plt.plot(x, y_fit, color='orange', linewidth=1,
                     label=r"Fit $y = A * e^{-x / B}$"
                           f"\nA: ({alpha:.2f} ± {err_alpha:.2f}) cm/s\n"
                           f"B: ({beta:.0f} ± {err_beta:.0f}) cm \n"
                           f"Chi2_rid: {chi2_red:.2f}")

            # plt.title(f"{nome}", pad=20)
            print(name)
            plt.legend()
            plt.grid()
            plt.show()

    if choice == 30:
        acceleration = []
        err_acceleration = []

        for name, set in Data.items():
            time_12 = set["1/2_time"].to_numpy()
            speed = set["speed"].to_numpy()
            err_time_12 = set["err_1/2_time"].to_numpy()
            err_speed = set["err_speed"].to_numpy()

            alpha, beta, err_alpha, err_beta, output = linear_fit(time_12, speed, err_x=err_time_12, err_y=err_speed)
            acceleration.append(alpha)
            err_acceleration.append(err_alpha)

        x = np.array(r)
        y = acceleration
        err_x = err_r
        err_y = err_acceleration

        plt.figure(figsize=(8, 6))
        plt.errorbar(x, y, xerr=err_x, yerr=err_y, fmt='o', capsize=5, capthick=1)
        plt.xlabel("raggio [mm]")
        plt.ylabel("accelerazione [cm/s^2]")

        '''alpha, beta, err_alpha, err_beta, output = linear_fit(x, y, err_x=err_x, err_y=err_y)
        y_fit = alpha * x + beta'''

        '''alpha, beta, gamma, err_alpha, err_beta, err_gamma, output = quadratic_fit(x, y, err_x=err_x, err_y=err_y)
        y_fit = alpha * x**2 + beta * x + gamma'''

        alpha, beta, gamma, epsilon, err_alpha, err_beta, err_gamma, err_epsilon, output = cubic_fit(x, y, err_x=err_x,
                                                                                                     err_y=err_y)
        y_fit = alpha * x ** 2 + beta * x + gamma

        chi2 = np.sum(((y - y_fit) / err_y) ** 2)
        dof = len(y) - 2
        chi2_red = chi2 / dof

        plt.plot(x, y_fit, color='orange', linewidth=1,
                 label=r"Fit $y = Ax + B$"
                       f"\nA: {alpha:.3f} ± {err_alpha:.3f}\n"
                       f"B: {beta:.2f} ± {err_beta:.2f}\n"
                       f"Chi2_red: {chi2_red:.2f}")
        plt.legend()

        plt.title("Fluido Newtoniano", pad=20)

        plt.grid()
        plt.show()


# def plot_histo(Time, choice):


if __name__ == "__main__":
    folder_path = r"C:\Users\loren\Desktop\viscosimetro\Ideale"
    dataframe = get_data(folder_path)
    Data = manipulate_data(dataframe)
    dataframe, Time = manipulate_results(dataframe, Data)
    Data = correction(Data)
    v_lim, err_v_lim = plot(Data, Time, dataframe, choice=1, head_skip=0, tail_skip=0)
    eta = calcolemus(v_lim, err_v_lim, Data, choice=True)
