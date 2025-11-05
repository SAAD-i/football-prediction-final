"""
League-specific views - North & Central America
"""
from django.shortcuts import render, get_object_or_404
from predictions.models import League, Team

def mls_usa(request):
    """MLS (USA) view"""
    league = get_object_or_404(League, slug='mls-usa', is_active=True)
    teams = Team.objects.filter(league=league).order_by('name')
    
    context = {
        'league': league,
        'teams': teams,
    }
    return render(request, 'predictions/leagues/mls_usa.html', context)

def mexican_liga_bbva_mx(request):
    """Mexican Liga BBVA MX view"""
    league = get_object_or_404(League, slug='mexican-liga-bbva-mx', is_active=True)
    teams = Team.objects.filter(league=league).order_by('name')
    
    context = {
        'league': league,
        'teams': teams,
    }
    return render(request, 'predictions/leagues/mexican_liga_bbva_mx.html', context)

