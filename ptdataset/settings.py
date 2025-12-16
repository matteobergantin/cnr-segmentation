from models.settings import ProjectSettings
import os

def _create_settings():
    if os.path.exists('settings.json'):
        return ProjectSettings.from_file('settings.json')
    else:
        r = ProjectSettings()
        r.save_to_file("settings.json")
        return r

settings: ProjectSettings = _create_settings()