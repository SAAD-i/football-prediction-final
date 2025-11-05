from django.urls import path
from . import views
from .views_europe_domestic_leagues import (
    english_premier_league, laliga_spain, italian_serie_a,
    german_bundesliga, french_ligue_1, portuguese_primeira_liga,
    scottish_premiership, efl_championship
)
from .views_europe_domestic_cups import (
    english_fa_cup, english_carabao_cup, spanish_copa_del_rey,
    german_cup, coppa_italia, coupe_de_france, scottish_league_cup
)
from .views_europe_uefa import (
    uefa_champions_league, uefa_europa_league, uefa_conference_league
)
from .views_international import (
    fifa_world_cup_qualification, fifa_world_cup_european_qualifiers,
    fifa_world_cup_asian_qualifiers
)
from .views_north_america import mls_usa, mexican_liga_bbva_mx
from .views_south_america import (
    conmebol_libertadores, argentine_liga_profesional, argentine_nacional_b,
    argentine_primera_c, brazilian_serie_b, brazilian_campeonato_paulista,
    brazilian_campeonato_gaucho, copa_do_brasil, copa_colombia,
    chilean_primera_division, ligapro_ecuador, liga_auf_uruguaya
)
from .views_asia import (
    indian_super_league, chinese_super_league, indonesian_super_league,
    singaporean_premier_league, thai_league_1, afc_champions_league_two
)
from .views_australia import australian_a_league_men

app_name = 'predictions'

urlpatterns = [
    path('', views.homepage, name='homepage'),
    path('api/predict/', views.predict_match, name='predict_match'),
    
    # Europe - Domestic Leagues
    path('league/english-premier-league/', english_premier_league, name='english_premier_league'),
    path('league/laliga-spain/', laliga_spain, name='laliga_spain'),
    path('league/italian-serie-a/', italian_serie_a, name='italian_serie_a'),
    path('league/german-bundesliga/', german_bundesliga, name='german_bundesliga'),
    path('league/french-ligue-1/', french_ligue_1, name='french_ligue_1'),
    path('league/portuguese-primeira-liga/', portuguese_primeira_liga, name='portuguese_primeira_liga'),
    path('league/scottish-premiership/', scottish_premiership, name='scottish_premiership'),
    path('league/efl-championship/', efl_championship, name='efl_championship'),
    
    # Europe - Domestic Cups
    path('league/english-fa-cup/', english_fa_cup, name='english_fa_cup'),
    path('league/english-carabao-cup/', english_carabao_cup, name='english_carabao_cup'),
    path('league/spanish-copa-del-rey/', spanish_copa_del_rey, name='spanish_copa_del_rey'),
    path('league/german-cup-dfb-pokal/', german_cup, name='german_cup'),
    path('league/coppa-italia/', coppa_italia, name='coppa_italia'),
    path('league/coupe-de-france/', coupe_de_france, name='coupe_de_france'),
    path('league/scottish-league-cup/', scottish_league_cup, name='scottish_league_cup'),
    
    # Europe - UEFA Competitions
    path('league/uefa-champions-league/', uefa_champions_league, name='uefa_champions_league'),
    path('league/uefa-europa-league/', uefa_europa_league, name='uefa_europa_league'),
    path('league/uefa-conference-league/', uefa_conference_league, name='uefa_conference_league'),
    
    # International - National Teams
    path('league/fifa-world-cup-qualification/', fifa_world_cup_qualification, name='fifa_world_cup_qualification'),
    path('league/fifa-world-cup-european-qualifiers/', fifa_world_cup_european_qualifiers, name='fifa_world_cup_european_qualifiers'),
    path('league/fifa-world-cup-asian-qualifiers/', fifa_world_cup_asian_qualifiers, name='fifa_world_cup_asian_qualifiers'),
    
    # North & Central America
    path('league/mls-usa/', mls_usa, name='mls_usa'),
    path('league/mexican-liga-bbva-mx/', mexican_liga_bbva_mx, name='mexican_liga_bbva_mx'),
    
    # South America
    path('league/conmebol-libertadores/', conmebol_libertadores, name='conmebol_libertadores'),
    path('league/argentine-liga-profesional/', argentine_liga_profesional, name='argentine_liga_profesional'),
    path('league/argentine-nacional-b/', argentine_nacional_b, name='argentine_nacional_b'),
    path('league/argentine-primera-c/', argentine_primera_c, name='argentine_primera_c'),
    path('league/brazilian-serie-b/', brazilian_serie_b, name='brazilian_serie_b'),
    path('league/brazilian-campeonato-paulista/', brazilian_campeonato_paulista, name='brazilian_campeonato_paulista'),
    path('league/brazilian-campeonato-gaucho/', brazilian_campeonato_gaucho, name='brazilian_campeonato_gaucho'),
    path('league/copa-do-brasil/', copa_do_brasil, name='copa_do_brasil'),
    path('league/copa-colombia/', copa_colombia, name='copa_colombia'),
    path('league/chilean-primera-division/', chilean_primera_division, name='chilean_primera_division'),
    path('league/ligapro-ecuador/', ligapro_ecuador, name='ligapro_ecuador'),
    path('league/liga-auf-uruguaya/', liga_auf_uruguaya, name='liga_auf_uruguaya'),
    
    # Asia
    path('league/indian-super-league/', indian_super_league, name='indian_super_league'),
    path('league/chinese-super-league/', chinese_super_league, name='chinese_super_league'),
    path('league/indonesian-super-league/', indonesian_super_league, name='indonesian_super_league'),
    path('league/singaporean-premier-league/', singaporean_premier_league, name='singaporean_premier_league'),
    path('league/thai-league-1/', thai_league_1, name='thai_league_1'),
    path('league/afc-champions-league-two/', afc_champions_league_two, name='afc_champions_league_two'),
    
    # Australia/Oceania
    path('league/australian-a-league-men/', australian_a_league_men, name='australian_a_league_men'),
]
