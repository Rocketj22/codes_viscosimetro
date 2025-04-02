Esperimento di Misura della Viscosità con Metodo delle Sfere
Questo repository contiene il codice Python utilizzato per analizzare i dati di un esperimento volto alla misura della viscosità di fluidi newtoniani e non-newtoniani attraverso il metodo della caduta di sfere, sfruttando la legge di Stokes.

Panoramica dell'Esperimento
L'esperimento consiste nella caduta di sfere di acciaio in cilindri contenenti due tipi di fluidi:
Glicerolo (fluido newtoniano)
Soluzione saponosa (fluido non-newtoniano)
Misurando tempo e spazio percorso dalle sfere, si ricava la velocità limite necessaria al calcolo della viscosità.

Struttura del Progetto
main.py: Script principale per l'analisi dati, contenente funzioni di lettura dati, manipolazione e calcolo della viscosità.

/dati: Directory contenente file sperimentali (formato .txt o .csv).

Dipendenze
Assicurarsi di avere installate le seguenti librerie Python:
pip install numpy pandas matplotlib scipy lmfit

Funzionalità del codice
Il codice consente di:
Leggere e gestire dati sperimentali.
Applicare modelli matematici (lineare, quadratico, cubico, esponenziale) per interpolare e analizzare i dati.
Effettuare correzioni degli effetti sistematici (effetto bordo).
Calcolare la viscosità tramite la propagazione degli errori.
Visualizzare graficamente risultati e fit statistici.

Come usare
Inserire i file di dati nella cartella dedicata (/dati).
Modificare, se necessario, il percorso nella variabile folder_path presente in fondo a main.py.

Eseguire lo script con:
python main.py

Risultati
Lo script genererà grafici dettagliati che mostrano:
Andamenti spazio-tempo e velocità-tempo.
Fit statistici e parametri fisici (viscosità).
Grafici dell'errore sperimentale e propagazione delle incertezze.

Autori
Nikolas Zuin
Lorenzo Molaro
Riccardo Maj

Supervisori: Prof. Michele Doro, Prof. Daniele Mengoni
Corso di Laurea Triennale in Fisica, Università degli Studi di Padova (A.A. 2024-2025)

