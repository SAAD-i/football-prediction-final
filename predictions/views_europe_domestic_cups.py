"""
League-specific views - Europe Domestic Cups
"""
from django.shortcuts import render, get_object_or_404
from predictions.models import League, Team

def english_fa_cup(request):
    """English FA Cup view"""
    league = get_object_or_404(League, slug='english-fa-cup', is_active=True)
    teams = Team.objects.filter(league=league).order_by('name')
    
    context = {
        'league': league,
        'teams': teams,
    }
    return render(request, 'predictions/leagues/english_fa_cup.html', context)

def english_carabao_cup(request):
    """English Carabao Cup view"""
    league = get_object_or_404(League, slug='english-carabao-cup', is_active=True)
    teams = Team.objects.filter(league=league).order_by('name')
    
    context = {
        'league': league,
        'teams': teams,
    }
    return render(request, 'predictions/leagues/english_carabao_cup.html', context)

def spanish_copa_del_rey(request):
    """Spanish Copa del Rey view"""
    league = get_object_or_404(League, slug='spanish-copa-del-rey', is_active=True)
    teams = Team.objects.filter(league=league).order_by('name')
    
    context = {
        'league': league,
        'teams': teams,
    }
    return render(request, 'predictions/leagues/spanish_copa_del_rey.html', context)

def german_cup(request):
    """German Cup (DFB-Pokal) view"""
    league = get_object_or_404(League, slug='german-cup-dfb-pokal', is_active=True)
    teams = Team.objects.filter(league=league).order_by('name')
    
    context = {
        'league': league,
        'teams': teams,
    }
    return render(request, 'predictions/leagues/german_cup.html', context)

def coppa_italia(request):
    """Coppa Italia view"""
    league = get_object_or_404(League, slug='coppa-italia', is_active=True)
    teams = Team.objects.filter(league=league).order_by('name')
    
    context = {
        'league': league,
        'teams': teams,
    }
    return render(request, 'predictions/leagues/coppa_italia.html', context)

def coupe_de_france(request):
    """Coupe de France view"""
    league = get_object_or_404(League, slug='coupe-de-france', is_active=True)
    teams = Team.objects.filter(league=league).order_by('name')
    
    context = {
        'league': league,
        'teams': teams,
    }
    return render(request, 'predictions/leagues/coupe_de_france.html', context)

def scottish_league_cup(request):
    """Scottish League Cup view"""
    league = get_object_or_404(League, slug='scottish-league-cup', is_active=True)
    teams = Team.objects.filter(league=league).order_by('name')
    
    context = {
        'league': league,
        'teams': teams,
    }
    return render(request, 'predictions/leagues/scottish_league_cup.html', context)

