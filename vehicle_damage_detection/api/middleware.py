# api/middleware.py

class PrintLinkMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

        # Afficher le lien à la création de l'instance du middleware
        print("\nLien d'accès à l'API de détection: 127.0.0.1:8000/api/detect/\n")

    def __call__(self, request):
        response = self.get_response(request)
        return response
