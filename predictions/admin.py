from django.contrib import admin
from .models import League, Team


@admin.register(League)
class LeagueAdmin(admin.ModelAdmin):
    list_display = ['name', 'category', 'is_active', 'slug']
    list_filter = ['category', 'is_active']
    search_fields = ['name', 'category']


@admin.register(Team)
class TeamAdmin(admin.ModelAdmin):
    list_display = ['name', 'league', 'slug']
    list_filter = ['league']
    search_fields = ['name', 'league__name']

