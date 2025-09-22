# Automatiserad neurala nätverksmodellering av havsisens utbredning i Norra ishavet

Det här projektet implementerar ett automatiserat flöde för att hämta kartbilder från MET Norge (https://api.met.no/weatherapi/icemap/1.0/documentation)
och tränar en maskininlärningsmodell (*Convolutional Neural Network*, CNN) att kunna uppskatta hur stor del av kartbilden som är täckt av is.

Observera att syftet med projektet är att skapa ett automatiserat flöde för att hämta data via API och träna om modeller.
Det behövs en stor mängd data (tusentals bilder) för att modellen ska kunna prestera bra.  
Programmet är kört så lång tid att ca 300 bilder har hämtats och tre modeller har tränats och sparats.

Detta är ett examinerande projekt i kursen *Fördjupning i python* vid EC Utbildning AB.

## Struktur
Projektet består av fyra delar
1. API
2. Databas
3. Maskininlärningsmodell
4. Huvudfunktionen

Kör programmet med
> python main.py  

För att träna en modell på befintlig data kör
> python model_utils.py

### 1. API-hämtning (api.py)
api.py innehåller en funktion för att hämta kartbilder från MET Norges REST API (https://api.met.no/weatherapi/icemap/1.0/documentation).
Tiden mellan requesten räknas autoamtiskt ut baserat på metadatan.

### 2. Databas (database_utils.py)
database_utils.py innehåller funktioner som skapar och hanterar en SQLite-databas där metadatan till bilderna sparas.
Metadatan består av tidpunkt för publicering och tidpunkt för nästa beräknade publicering samt lokal sökväg till den sparade bilden.

### 3. Maskininlärningsmodell (model_util.py)
model_utils.py innehåller en klass med en CNN-modell samt funktioner för att träna och spara modeller.
Datan delas upp i tränings-, validerings- och testdel. Modellen tränas med MSE som loss-funktion.
Varje modell som tränats sparas separat och jämförs med varandra. Den modell med lägst valdieringsloss sparas som *best_model*.

### 4. Huvudfunktionen
main-py innehåller huvudfunktionen main som initierar databasen och startar en loop som skickar
request till API:et samt tränar och sparar en modell vid ett intervall av 100 bilder.

**Sissel Nevestveit - september 2025**