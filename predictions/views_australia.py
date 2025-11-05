"""
League-specific views - Australia/Oceania
"""
from django.shortcuts import render, get_object_or_404
from predictions.models import League, Team

def australian_a_league_men(request):
    """Australian A-League Men view"""
    league = get_object_or_404(League, slug='australian-a-league-men', is_active=True)
    teams = Team.objects.filter(league=league).order_by('name')
    
    context = {
        'league': league,
        'teams': teams,
    }
    return render(request, 'predictions/leagues/australian_a_league_men.html', context)

