"""
League-specific views - Asia
"""
from django.shortcuts import render, get_object_or_404
from predictions.models import League, Team

def indian_super_league(request):
    """Indian Super League view"""
    league = get_object_or_404(League, slug='indian-super-league', is_active=True)
    teams = Team.objects.filter(league=league).order_by('name')
    
    context = {
        'league': league,
        'teams': teams,
    }
    return render(request, 'predictions/leagues/indian_super_league.html', context)

def chinese_super_league(request):
    """Chinese Super League view"""
    league = get_object_or_404(League, slug='chinese-super-league', is_active=True)
    teams = Team.objects.filter(league=league).order_by('name')
    
    context = {
        'league': league,
        'teams': teams,
    }
    return render(request, 'predictions/leagues/chinese_super_league.html', context)

def indonesian_super_league(request):
    """Indonesian Super League view"""
    league = get_object_or_404(League, slug='indonesian-super-league', is_active=True)
    teams = Team.objects.filter(league=league).order_by('name')
    
    context = {
        'league': league,
        'teams': teams,
    }
    return render(request, 'predictions/leagues/indonesian_super_league.html', context)

def singaporean_premier_league(request):
    """Singaporean Premier League view"""
    league = get_object_or_404(League, slug='singaporean-premier-league', is_active=True)
    teams = Team.objects.filter(league=league).order_by('name')
    
    context = {
        'league': league,
        'teams': teams,
    }
    return render(request, 'predictions/leagues/singaporean_premier_league.html', context)

def thai_league_1(request):
    """Thai League 1 view"""
    league = get_object_or_404(League, slug='thai-league-1', is_active=True)
    teams = Team.objects.filter(league=league).order_by('name')
    
    context = {
        'league': league,
        'teams': teams,
    }
    return render(request, 'predictions/leagues/thai_league_1.html', context)

def afc_champions_league_two(request):
    """AFC Champions League Two view"""
    league = get_object_or_404(League, slug='afc-champions-league-two', is_active=True)
    teams = Team.objects.filter(league=league).order_by('name')
    
    context = {
        'league': league,
        'teams': teams,
    }
    return render(request, 'predictions/leagues/afc_champions_league_two.html', context)

