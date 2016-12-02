from django.shortcuts import render
from rest_framework.decorators import api_view
from rest_framework.response import Response
import utils.assert_food


@api_view(['get'])
def assert_food(img_url):
    image = '/home/chong/Pictures/food/1.jpg'
    pred = utils.assert_food.run_inference_on_image(image)
    rst = utils.assert_food.assert_food(pred)
    return Response({'data': rst})
