from models.settings import ProjectSettings
import os

settings: ProjectSettings
if os.path.exists('settings.json'):
    settings = ProjectSettings.from_file('settings.json')
else:
    settings = ProjectSettings()