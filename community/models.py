from django.db import models

class FreeBoard(models.Model):
    free_board_id = models.AutoField(primary_key=True)  # 기본키: 자동 증가
    user = models.ForeignKey('account.User', on_delete=models.CASCADE, related_name='free_boards')  # 외래키: account.User 모델 참조
    title = models.CharField(max_length=200)  # 제목, 최대 200자
    content = models.TextField()  # 내용
    reg_dt = models.DateTimeField(auto_now_add=True)  # 등록일, 자동 생성
    mod_dt = models.DateTimeField(auto_now=True)  # 수정일, 자동 갱신
    is_deleted = models.BooleanField(default=False)  # 삭제 여부, 기본값 False

    class Meta:
        db_table = 'free_board'  # 테이블 이름 지정
        ordering = ['-reg_dt']  # 최신 게시물 먼저 정렬

    def __str__(self):
        return self.title