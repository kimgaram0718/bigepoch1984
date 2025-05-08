from django.db import models
from account.models import User

from django.db import models

# ... (기존 다른 모델 정의들) ...

class Disclosure(models.Model):
    disclosure_type = models.CharField(max_length=50)   # 공시 종류 (예: 정기공시, 주요사항보고 등)
    date = models.DateField(null=True, blank=True)                          # 공시 날짜 (접수일자)
    title = models.CharField(max_length=255)           # 공시 제목
    content = models.TextField()                       # 공시 본문 내용 (텍스트)

    def __str__(self):
        return f"[{self.disclosure_type}] {self.title}"
class FreeBoard(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    title = models.CharField(max_length=200)
    content = models.TextField()
    reg_dt = models.DateTimeField(auto_now_add=True)
    is_deleted = models.BooleanField(default=False)
    likes_count = models.PositiveIntegerField(default=0)  # 좋아요 수
    comments_count = models.PositiveIntegerField(default=0)  # 댓글 수

    class Meta:
        db_table = 'free_board'

class FreeBoardLike(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    free_board = models.ForeignKey(FreeBoard, on_delete=models.CASCADE, related_name='likes')
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = 'free_board_like'
        unique_together = ('user', 'free_board')  # 사용자당 게시글에 1번만 좋아요

class FreeBoardComment(models.Model):
    free_board = models.ForeignKey(FreeBoard, on_delete=models.CASCADE, related_name='comments')
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    content = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)
    is_deleted = models.BooleanField(default=False)

    class Meta:
        db_table = 'free_board_comment'
        ordering = ['created_at']