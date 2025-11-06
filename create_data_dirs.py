"""Create directory structure for CSV files"""
from pathlib import Path

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / 'data' / 'Europe-Domestic-Leagues'

# League directories to create
leagues = [
    'EPL',
    'LaLiga-Spain',
    'Italian-Serie-A',
    'German-Bundesliga',
    'French-Ligue-1',
    'Portuguese-Primeira-Liga',
    'EFL-Championship',
    'Scottish-Premiership',
]

# Create all directories
for league in leagues:
    league_dir = DATA_DIR / league
    league_dir.mkdir(parents=True, exist_ok=True)
    print(f"Created: {league_dir}")

print(f"\nAll directories created in: {DATA_DIR}")
print("\nNow copy your CSV files:")
print("  - EPL/epldata.csv")
print("  - LaLiga-Spain/laligadata.csv")
print("  - Italian-Serie-A/serieadata.csv")
print("  - German-Bundesliga/bundesligadata.csv")
print("  - French-Ligue-1/ligue1data.csv")
print("  - Portuguese-Primeira-Liga/primeiraligadata.csv")
print("  - EFL-Championship/championshipdata.csv")
print("  - Scottish-Premiership/scottishpremdata.csv")

