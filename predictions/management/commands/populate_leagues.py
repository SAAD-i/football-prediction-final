from django.core.management.base import BaseCommand
from predictions.models import League, Team


class Command(BaseCommand):
    help = 'Populate leagues and teams data'

    def handle(self, *args, **options):
        # Define all leagues
        leagues_data = [
            # Europe – Domestic Leagues
            {'name': 'English Premier League', 'category': 'Europe – Domestic Leagues', 'slug': 'english-premier-league', 'model_path': 'epl'},
            {'name': 'LaLiga (Spain)', 'category': 'Europe – Domestic Leagues', 'slug': 'laliga-spain'},
            {'name': 'Italian Serie A', 'category': 'Europe – Domestic Leagues', 'slug': 'italian-serie-a'},
            {'name': 'German Bundesliga', 'category': 'Europe – Domestic Leagues', 'slug': 'german-bundesliga'},
            {'name': 'French Ligue 1', 'category': 'Europe – Domestic Leagues', 'slug': 'french-ligue-1'},
            {'name': 'Portuguese Primeira Liga', 'category': 'Europe – Domestic Leagues', 'slug': 'portuguese-primeira-liga'},
            {'name': 'Scottish Premiership', 'category': 'Europe – Domestic Leagues', 'slug': 'scottish-premiership'},
            {'name': 'EFL Championship', 'category': 'Europe – Domestic Leagues', 'slug': 'efl-championship'},
            
            # Europe – Domestic Cups
            {'name': 'English FA Cup', 'category': 'Europe – Domestic Cups', 'slug': 'english-fa-cup'},
            {'name': 'English Carabao Cup (EFL Cup)', 'category': 'Europe – Domestic Cups', 'slug': 'english-carabao-cup'},
            {'name': 'Spanish Copa del Rey', 'category': 'Europe – Domestic Cups', 'slug': 'spanish-copa-del-rey'},
            {'name': 'German Cup (DFB-Pokal)', 'category': 'Europe – Domestic Cups', 'slug': 'german-cup-dfb-pokal'},
            {'name': 'Coppa Italia', 'category': 'Europe – Domestic Cups', 'slug': 'coppa-italia'},
            {'name': 'Coupe de France', 'category': 'Europe – Domestic Cups', 'slug': 'coupe-de-france'},
            {'name': 'Scottish League Cup', 'category': 'Europe – Domestic Cups', 'slug': 'scottish-league-cup'},
            
            # Europe – UEFA Competitions
            {'name': 'UEFA Champions League', 'category': 'Europe – UEFA Competitions', 'slug': 'uefa-champions-league'},
            {'name': 'UEFA Europa League', 'category': 'Europe – UEFA Competitions', 'slug': 'uefa-europa-league'},
            {'name': 'UEFA Conference League', 'category': 'Europe – UEFA Competitions', 'slug': 'uefa-conference-league'},
            
            # International – National Teams
            {'name': 'FIFA World Cup qualification', 'category': 'International – National Teams', 'slug': 'fifa-world-cup-qualification'},
            {'name': 'FIFA World Cup European qualifiers', 'category': 'International – National Teams', 'slug': 'fifa-world-cup-european-qualifiers'},
            {'name': 'FIFA World Cup Asian qualifiers', 'category': 'International – National Teams', 'slug': 'fifa-world-cup-asian-qualifiers'},
            
            # North & Central America
            {'name': 'MLS (USA)', 'category': 'North & Central America', 'slug': 'mls-usa'},
            {'name': 'Mexican Liga BBVA MX', 'category': 'North & Central America', 'slug': 'mexican-liga-bbva-mx'},
            
            # South America
            {'name': 'CONMEBOL Libertadores', 'category': 'South America', 'slug': 'conmebol-libertadores'},
            {'name': 'Argentine Liga Profesional de Fútbol', 'category': 'South America', 'slug': 'argentine-liga-profesional'},
            {'name': 'Argentine Nacional B', 'category': 'South America', 'slug': 'argentine-nacional-b'},
            {'name': 'Argentine Primera C', 'category': 'South America', 'slug': 'argentine-primera-c'},
            {'name': 'Brazilian Serie B', 'category': 'South America', 'slug': 'brazilian-serie-b'},
            {'name': 'Brazilian Campeonato Paulista', 'category': 'South America', 'slug': 'brazilian-campeonato-paulista'},
            {'name': 'Brazilian Campeonato Gaucho', 'category': 'South America', 'slug': 'brazilian-campeonato-gaucho'},
            {'name': 'Copa do Brasil', 'category': 'South America', 'slug': 'copa-do-brasil'},
            {'name': 'Copa Colombia', 'category': 'South America', 'slug': 'copa-colombia'},
            {'name': 'Chilean Primera División', 'category': 'South America', 'slug': 'chilean-primera-division'},
            {'name': 'LigaPro Ecuador', 'category': 'South America', 'slug': 'ligapro-ecuador'},
            {'name': 'Liga AUF Uruguaya', 'category': 'South America', 'slug': 'liga-auf-uruguaya'},
            
            # Asia
            {'name': 'Indian Super League', 'category': 'Asia', 'slug': 'indian-super-league'},
            {'name': 'Chinese Super League', 'category': 'Asia', 'slug': 'chinese-super-league'},
            {'name': 'Indonesian Super League', 'category': 'Asia', 'slug': 'indonesian-super-league'},
            {'name': 'Singaporean Premier League', 'category': 'Asia', 'slug': 'singaporean-premier-league'},
            {'name': 'Thai League 1', 'category': 'Asia', 'slug': 'thai-league-1'},
            {'name': 'AFC Champions League Two', 'category': 'Asia', 'slug': 'afc-champions-league-two'},
            
            # Australia/Oceania
            {'name': 'Australian A-League Men', 'category': 'Australia/Oceania', 'slug': 'australian-a-league-men'},
        ]
        
        # EPL teams (from preprocessing_parameters.json)
        epl_teams = [
            "Arsenal", "Aston Villa", "Birmingham", "Blackburn", "Blackpool", "Bolton",
            "Bournemouth", "Bradford", "Brentford", "Brighton", "Burnley", "Cardiff",
            "Charlton", "Chelsea", "Coventry", "Crystal Palace", "Derby", "Everton",
            "Fulham", "Huddersfield", "Hull", "Ipswich", "Leeds", "Leicester",
            "Liverpool", "Luton", "Man City", "Man United", "Middlesboro", "Middlesbrough",
            "Newcastle", "Norwich", "Nott'm Forest", "Portsmouth", "QPR", "Reading",
            "Sheffield United", "Sheffield Weds", "Southampton", "Stoke", "Sunderland",
            "Swansea", "Tottenham", "Watford", "West Brom", "West Ham", "Wigan",
            "Wimbledon", "Wolves"
        ]
        
        # Create leagues
        for league_data in leagues_data:
            league, created = League.objects.get_or_create(
                slug=league_data['slug'],
                defaults={
                    'name': league_data['name'],
                    'category': league_data['category'],
                    'is_active': True,
                }
            )
            if created:
                self.stdout.write(self.style.SUCCESS(f'Created league: {league.name}'))
            else:
                self.stdout.write(f'League already exists: {league.name}')
            
            # Add EPL teams
            if league.slug == 'english-premier-league':
                for team_name in epl_teams:
                    team, team_created = Team.objects.get_or_create(
                        league=league,
                        name=team_name,
                        defaults={'slug': team_name.lower().replace(' ', '-').replace("'", '').replace('&', 'and')}
                    )
                    if team_created:
                        self.stdout.write(f'  Created team: {team_name}')
        
        self.stdout.write(self.style.SUCCESS('\nSuccessfully populated leagues and teams!'))

