from django.shortcuts import render
from rest_framework.decorators import api_view
from rest_framework.response import Response
import utils.assert_food


@api_view(['post'])
def assert_food(request):
    image_url = request.POST.get('img_url')
    pred = utils.assert_food.run_inference_on_image(image_url)
    rst = utils.assert_food.assert_food(pred)
    return Response({'data': rst})
