from django.shortcuts import render

def index_view(request):
    """
    Renders the main index page.
    """
    # In the future, you might pass context data to the template here
    context = {}
    return render(request, 'index.html', context)

# Add other views for handling form submission, etc., later