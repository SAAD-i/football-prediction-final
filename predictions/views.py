from django.shortcuts import render, redirect, get_object_or_404
from django.http import JsonResponse
from django.views.decorators.http import require_http_methods
from django.views.decorators.csrf import csrf_exempt
import json

from .models import League, Team
from .services import ONNXPredictor


def homepage(request):
    """Homepage displaying all leagues"""
    leagues_by_category = {}
    
    try:
        # Group leagues by category
        leagues = League.objects.filter(is_active=True).order_by('category', 'name')
        for league in leagues:
            category = league.category
            if category not in leagues_by_category:
                leagues_by_category[category] = []
            leagues_by_category[category].append(league)
    except Exception as e:
        # Handle database errors gracefully
        import traceback
        traceback.print_exc()
        # Return empty context if database is not ready
        pass
    
    context = {
        'leagues_by_category': leagues_by_category,
    }
    return render(request, 'predictions/homepage.html', context)


def league_detail(request, slug):
    """Display league details and team selection"""
    league = get_object_or_404(League, slug=slug, is_active=True)
    teams = Team.objects.filter(league=league).order_by('name')
    
    # Use league-specific templates
    template_map = {
        'epl': 'predictions/epl.html',
        'english-premier-league': 'predictions/epl.html',
    }
    
    template_name = template_map.get(slug, 'predictions/league_detail.html')
    
    context = {
        'league': league,
        'teams': teams,
    }
    return render(request, template_name, context)


@csrf_exempt
@require_http_methods(["POST"])
def predict_match(request):
    """Predict match outcome using ONNX model"""
    try:
        data = json.loads(request.body)
        league_slug = data.get('league_slug')
        home_team = data.get('home_team')
        away_team = data.get('away_team')
        
        if not all([league_slug, home_team, away_team]):
            return JsonResponse({
                'success': False,
                'error': 'Missing required fields: league_slug, home_team, away_team'
            }, status=400)
        
        # Get league
        league = get_object_or_404(League, slug=league_slug, is_active=True)
        
        # Check if model is available - handle both slugs
        if league_slug not in ['epl', 'english-premier-league']:
            return JsonResponse({
                'success': False,
                'error': f'Prediction not yet available for {league.name}. Only EPL is currently supported.'
            }, status=400)
        
        # Normalize league slug for predictor
        predictor_slug = 'epl' if league_slug in ['epl', 'english-premier-league'] else league_slug
        
        # Make prediction
        predictor = ONNXPredictor(predictor_slug)
        predicted, probabilities = predictor.predict(home_team, away_team)
        
        # Format response
        outcome_map = {
            'H': 'Home Win',
            'D': 'Draw',
            'A': 'Away Win'
        }
        
        response_data = {
            'success': True,
            'prediction': {
                'outcome': predicted,
                'outcome_label': outcome_map.get(predicted, predicted),
                'probabilities': {
                    'home_win': probabilities.get('H', 0.0),
                    'draw': probabilities.get('D', 0.0),
                    'away_win': probabilities.get('A', 0.0),
                }
            }
        }
        
        return JsonResponse(response_data)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JsonResponse({
            'success': False,
            'error': str(e)
        }, status=500)

