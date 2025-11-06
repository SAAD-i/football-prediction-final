from django.shortcuts import render, redirect, get_object_or_404
from django.http import JsonResponse
from django.views.decorators.http import require_http_methods
from django.views.decorators.csrf import csrf_exempt
import json

from .models import League, Team
# Don't import ONNXPredictor here - lazy import in predict_match to avoid homepage errors


class StaticLeague:
    """Simple class to represent a league with static data"""
    def __init__(self, name, category, slug, model_path=None):
        self.name = name
        self.category = category
        self.slug = slug
        self.model_path = model_path


def homepage(request):
    """Homepage displaying all leagues - completely static HTML"""
    # Return static template - no context needed
    return render(request, 'predictions/homepage.html')


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
        # Lazy import to avoid import errors affecting homepage
        from .services import ONNXPredictor
        
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
        
        # Supported leagues (Europe - Domestic Leagues and Cups)
        supported_leagues = [
            'epl', 'english-premier-league',
            'laliga-spain',
            'italian-serie-a',
            'german-bundesliga',
            'french-ligue-1',
            'portuguese-primeira-liga',
            'efl-championship',
            'scottish-premiership',
            # Europe - Domestic Cups
            'english-fa-cup',
            'english-carabao-cup',
            'spanish-copa-del-rey',
            'german-cup-dfb-pokal',
            'coppa-italia',
            'coupe-de-france',
            'scottish-league-cup',
        ]
        
        # Check if model is available
        if league_slug not in supported_leagues:
            return JsonResponse({
                'success': False,
                'error': f'Prediction not yet available for {league.name}. Currently supported: {", ".join(supported_leagues)}'
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

