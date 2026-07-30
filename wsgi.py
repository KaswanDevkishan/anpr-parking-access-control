"""Production WSGI entry point."""

from anpr_web import create_app

app = create_app()
