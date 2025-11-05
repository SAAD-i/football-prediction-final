from django.db import models


class League(models.Model):
    """Model for football leagues"""
    name = models.CharField(max_length=200)
    category = models.CharField(max_length=100)  # e.g., "Europe – Domestic Leagues"
    slug = models.SlugField(unique=True)
    is_active = models.BooleanField(default=True)
    model_path = models.CharField(max_length=500, blank=True, null=True)  # Path to ONNX model
    preprocessing_path = models.CharField(max_length=500, blank=True, null=True)  # Path to preprocessing JSON
    
    class Meta:
        ordering = ['category', 'name']
    
    def __str__(self):
        return self.name


class Team(models.Model):
    """Model for football teams"""
    league = models.ForeignKey(League, on_delete=models.CASCADE, related_name='teams')
    name = models.CharField(max_length=200)
    slug = models.SlugField()
    
    class Meta:
        unique_together = ['league', 'name']
        ordering = ['name']
    
    def __str__(self):
        return f"{self.name} ({self.league.name})"

