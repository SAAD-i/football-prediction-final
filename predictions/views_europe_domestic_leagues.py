"""
League-specific views - Europe Domestic Leagues
"""
from django.shortcuts import render, get_object_or_404
from predictions.models import League, Team

def english_premier_league(request):
    """English Premier League view"""
    league = get_object_or_404(League, slug='english-premier-league', is_active=True)
    teams = Team.objects.filter(league=league).order_by('name')
    
    context = {
        'league': league,
        'teams': teams,
    }
    return render(request, 'predictions/leagues/english_premier_league.html', context)

def laliga_spain(request):
    """LaLiga (Spain) view"""
    league = get_object_or_404(League, slug='laliga-spain', is_active=True)
    teams = Team.objects.filter(league=league).order_by('name')
    
    context = {
        'league': league,
        'teams': teams,
    }
    return render(request, 'predictions/leagues/laliga_spain.html', context)

def italian_serie_a(request):
    """Italian Serie A view"""
    league = get_object_or_404(League, slug='italian-serie-a', is_active=True)
    teams = Team.objects.filter(league=league).order_by('name')
    
    context = {
        'league': league,
        'teams': teams,
    }
    return render(request, 'predictions/leagues/italian_serie_a.html', context)

def german_bundesliga(request):
    """German Bundesliga view"""
    league = get_object_or_404(League, slug='german-bundesliga', is_active=True)
    teams = Team.objects.filter(league=league).order_by('name')
    
    context = {
        'league': league,
        'teams': teams,
    }
    return render(request, 'predictions/leagues/german_bundesliga.html', context)

def french_ligue_1(request):
    """French Ligue 1 view"""
    league = get_object_or_404(League, slug='french-ligue-1', is_active=True)
    teams = Team.objects.filter(league=league).order_by('name')
    
    context = {
        'league': league,
        'teams': teams,
    }
    return render(request, 'predictions/leagues/french_ligue_1.html', context)

def portuguese_primeira_liga(request):
    """Portuguese Primeira Liga view"""
    league = get_object_or_404(League, slug='portuguese-primeira-liga', is_active=True)
    teams = Team.objects.filter(league=league).order_by('name')
    
    context = {
        'league': league,
        'teams': teams,
    }
    return render(request, 'predictions/leagues/portuguese_primeira_liga.html', context)

def scottish_premiership(request):
    """Scottish Premiership view"""
    league = get_object_or_404(League, slug='scottish-premiership', is_active=True)
    teams = Team.objects.filter(league=league).order_by('name')
    if not teams.exists():
        # Fallback: load team names from preprocessing_parameters.json
        import json
        from pathlib import Path
        from django.conf import settings
        json_path = Path(settings.BASE_DIR) / 'predictions' / 'models_storage' / 'Europe-Domestic-Leagues' / 'ScotishPremiership' / 'preprocessing_parameters.json'
        fallback_teams = []
        try:
            if json_path.exists():
                with open(json_path, 'r') as f:
                    data = json.load(f)
                    team_names = data.get('teams', [])
                    fallback_teams = [{'name': n} for n in team_names]
        except Exception:
            fallback_teams = []
    
    context = {
        'league': league,
        'teams': teams if teams.exists() else fallback_teams,
    }
    return render(request, 'predictions/leagues/scottish_premiership.html', context)

def efl_championship(request):
    """EFL Championship view"""
    league = get_object_or_404(League, slug='efl-championship', is_active=True)
    teams = Team.objects.filter(league=league).order_by('name')
    
    context = {
        'league': league,
        'teams': teams,
    }
    return render(request, 'predictions/leagues/efl_championship.html', context)


