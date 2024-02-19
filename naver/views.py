from django.shortcuts import render
import requests

# Create your views here.
def search_panorama(request):
    if request.method == 'POST':
        address = request.POST.get('address')  # POST로부터 검색할 주소를 가져옴
        return render(request, 'panorama.html', {'address': address})
    else:
        return render(request, 'search.html', {'error_message': '잘못된 접근입니다.'})