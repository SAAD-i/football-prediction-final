"""
League-specific views - South America
"""
from django.shortcuts import render, get_object_or_404
from predictions.models import League, Team

def conmebol_libertadores(request):
    """CONMEBOL Libertadores view"""
    league = get_object_or_404(League, slug='conmebol-libertadores', is_active=True)
    teams = Team.objects.filter(league=league).order_by('name')
    
    context = {
        'league': league,
        'teams': teams,
    }
    return render(request, 'predictions/leagues/conmebol_libertadores.html', context)

def argentine_liga_profesional(request):
    """Argentine Liga Profesional de Fútbol view"""
    league = get_object_or_404(League, slug='argentine-liga-profesional', is_active=True)
    teams = Team.objects.filter(league=league).order_by('name')
    
    context = {
        'league': league,
        'teams': teams,
    }
    return render(request, 'predictions/leagues/argentine_liga_profesional.html', context)

def argentine_nacional_b(request):
    """Argentine Nacional B view"""
    league = get_object_or_404(League, slug='argentine-nacional-b', is_active=True)
    teams = Team.objects.filter(league=league).order_by('name')
    
    context = {
        'league': league,
        'teams': teams,
    }
    return render(request, 'predictions/leagues/argentine_nacional_b.html', context)

def argentine_primera_c(request):
    """Argentine Primera C view"""
    league = get_object_or_404(League, slug='argentine-primera-c', is_active=True)
    teams = Team.objects.filter(league=league).order_by('name')
    
    context = {
        'league': league,
        'teams': teams,
    }
    return render(request, 'predictions/leagues/argentine_primera_c.html', context)

def brazilian_serie_b(request):
    """Brazilian Serie B view"""
    league = get_object_or_404(League, slug='brazilian-serie-b', is_active=True)
    teams = Team.objects.filter(league=league).order_by('name')
    
    context = {
        'league': league,
        'teams': teams,
    }
    return render(request, 'predictions/leagues/brazilian_serie_b.html', context)

def brazilian_campeonato_paulista(request):
    """Brazilian Campeonato Paulista view"""
    league = get_object_or_404(League, slug='brazilian-campeonato-paulista', is_active=True)
    teams = Team.objects.filter(league=league).order_by('name')
    
    context = {
        'league': league,
        'teams': teams,
    }
    return render(request, 'predictions/leagues/brazilian_campeonato_paulista.html', context)

def brazilian_campeonato_gaucho(request):
    """Brazilian Campeonato Gaucho view"""
    league = get_object_or_404(League, slug='brazilian-campeonato-gaucho', is_active=True)
    teams = Team.objects.filter(league=league).order_by('name')
    
    context = {
        'league': league,
        'teams': teams,
    }
    return render(request, 'predictions/leagues/brazilian_campeonato_gaucho.html', context)

def copa_do_brasil(request):
    """Copa do Brasil view"""
    league = get_object_or_404(League, slug='copa-do-brasil', is_active=True)
    teams = Team.objects.filter(league=league).order_by('name')
    
    context = {
        'league': league,
        'teams': teams,
    }
    return render(request, 'predictions/leagues/copa_do_brasil.html', context)

def copa_colombia(request):
    """Copa Colombia view"""
    league = get_object_or_404(League, slug='copa-colombia', is_active=True)
    teams = Team.objects.filter(league=league).order_by('name')
    
    context = {
        'league': league,
        'teams': teams,
    }
    return render(request, 'predictions/leagues/copa_colombia.html', context)

def chilean_primera_division(request):
    """Chilean Primera División view"""
    league = get_object_or_404(League, slug='chilean-primera-division', is_active=True)
    teams = Team.objects.filter(league=league).order_by('name')
    
    context = {
        'league': league,
        'teams': teams,
    }
    return render(request, 'predictions/leagues/chilean_primera_division.html', context)

def ligapro_ecuador(request):
    """LigaPro Ecuador view"""
    league = get_object_or_404(League, slug='ligapro-ecuador', is_active=True)
    teams = Team.objects.filter(league=league).order_by('name')
    
    context = {
        'league': league,
        'teams': teams,
    }
    return render(request, 'predictions/leagues/ligapro_ecuador.html', context)

def liga_auf_uruguaya(request):
    """Liga AUF Uruguaya view"""
    league = get_object_or_404(League, slug='liga-auf-uruguaya', is_active=True)
    teams = Team.objects.filter(league=league).order_by('name')
    
    context = {
        'league': league,
        'teams': teams,
    }
    return render(request, 'predictions/leagues/liga_auf_uruguaya.html', context)

