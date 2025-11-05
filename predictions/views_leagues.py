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

