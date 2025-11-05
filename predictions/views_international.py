"""
League-specific views - International National Teams
"""
from django.shortcuts import render, get_object_or_404
from predictions.models import League, Team

def fifa_world_cup_qualification(request):
    """FIFA World Cup qualification view"""
    league = get_object_or_404(League, slug='fifa-world-cup-qualification', is_active=True)
    teams = Team.objects.filter(league=league).order_by('name')
    
    context = {
        'league': league,
        'teams': teams,
    }
    return render(request, 'predictions/leagues/fifa_world_cup_qualification.html', context)

def fifa_world_cup_european_qualifiers(request):
    """FIFA World Cup European qualifiers view"""
    league = get_object_or_404(League, slug='fifa-world-cup-european-qualifiers', is_active=True)
    teams = Team.objects.filter(league=league).order_by('name')
    
    context = {
        'league': league,
        'teams': teams,
    }
    return render(request, 'predictions/leagues/fifa_world_cup_european_qualifiers.html', context)

def fifa_world_cup_asian_qualifiers(request):
    """FIFA World Cup Asian qualifiers view"""
    league = get_object_or_404(League, slug='fifa-world-cup-asian-qualifiers', is_active=True)
    teams = Team.objects.filter(league=league).order_by('name')
    
    context = {
        'league': league,
        'teams': teams,
    }
    return render(request, 'predictions/leagues/fifa_world_cup_asian_qualifiers.html', context)

