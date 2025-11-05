"""
League-specific views - Europe UEFA Competitions
"""
from django.shortcuts import render, get_object_or_404
from predictions.models import League, Team

def uefa_champions_league(request):
    """UEFA Champions League view"""
    league = get_object_or_404(League, slug='uefa-champions-league', is_active=True)
    teams = Team.objects.filter(league=league).order_by('name')
    
    context = {
        'league': league,
        'teams': teams,
    }
    return render(request, 'predictions/leagues/uefa_champions_league.html', context)

def uefa_europa_league(request):
    """UEFA Europa League view"""
    league = get_object_or_404(League, slug='uefa-europa-league', is_active=True)
    teams = Team.objects.filter(league=league).order_by('name')
    
    context = {
        'league': league,
        'teams': teams,
    }
    return render(request, 'predictions/leagues/uefa_europa_league.html', context)

def uefa_conference_league(request):
    """UEFA Conference League view"""
    league = get_object_or_404(League, slug='uefa-conference-league', is_active=True)
    teams = Team.objects.filter(league=league).order_by('name')
    
    context = {
        'league': league,
        'teams': teams,
    }
    return render(request, 'predictions/leagues/uefa_conference_league.html', context)

