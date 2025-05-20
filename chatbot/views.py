from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
from openai import OpenAI

# OpenAI 클라이언트 초기화
client = OpenAI(api_key="")

def chatbot_page(request):
    return render(request, 'chatbot.html')

@csrf_exempt
def chat_response(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            user_message = data.get('message', '')
            
            print(f"Received message: {user_message}")  # 디버깅용 로그
            
            try:
                # OpenAI API 호출
                response = client.chat.completions.create(
                    model="gpt-4-turbo",
                    messages=[
                        {"role": "system", "content": (
                            "당신은 금융 전문 어시스턴트입니다. "
                            "주식, 투자, 경제, 금융상품, 부동산, 재테크, 가계재무, 금융시장 등 금융과 관련된 모든 주제에 대해 전문적이고 실용적인 답변을 제공합니다. "
                            "금융과 무관한 질문(예: 요리, 여행, 게임 등)에는 '죄송합니다만, 저는 금융 관련 문의사항에 대해서만 도움을 드릴 수 있습니다.'라고 답변하세요. "
                            "모든 답변은 한국어로, 간결하고 실용적으로, 사용자가 이해하기 쉽게 설명하세요. "
                            "질문에 대한 답변만 출력하세요. 불필요한 포맷(A(고객사):, B(챗봇): 등)은 붙이지 마세요."
                        )},
                        {"role": "user", "content": user_message}
                    ],
                    temperature=0.7,
                    max_tokens=1000
                )
                
                # API 응답에서 챗봇 메시지 추출
                bot_response = response.choices[0].message.content
                print(f"OpenAI 응답: {repr(bot_response)}")  # 실제 응답을 repr로 출력
                
                return JsonResponse({
                    'status': 'success',
                    'message': bot_response
                })
                
            except Exception as api_error:
                print(f"OpenAI API Error: {str(api_error)}")  # 디버깅용 로그
                return JsonResponse({
                    'status': 'error',
                    'message': f"OpenAI API 오류: {str(api_error)}"
                }, status=500)

        except Exception as e:
            print(f"General Error: {str(e)}")  # 디버깅용 로그
            return JsonResponse({
                'status': 'error',
                'message': f"서버 오류가 발생했습니다: {str(e)}"
            }, status=500)

    return JsonResponse({'status': 'error', 'message': 'Invalid request method'}, status=405)