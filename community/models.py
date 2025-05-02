from django.db import models
from account.models import User

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