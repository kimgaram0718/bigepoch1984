from django.db import models

# Create your models here.
class User(models.Model):
    AUTH_CHOICES = [
        ('admin', '운영자'),
        ('user', '일반회원'),
    ]
    user_id = models.AutoField(primary_key=True)
    login_id = models.CharField(max_length=50, unique=True)
    auth_id = models.CharField(max_length=10, choices=AUTH_CHOICES, default='user')
    pwd = models.CharField(max_length=128)
    nickname = models.CharField(max_length=50, unique=True)
    email = models.EmailField(unique=True)
    created_at = models.DateTimeField(auto_now_add=True)  # 생성 시간

    def __str__(self):
        return self.login_id