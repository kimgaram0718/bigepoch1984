from django.db import models

# Create your models here.

class StockPrice(models.Model):
    market = models.CharField(max_length=10)  # 'KOSPI' 또는 'KOSDAQ'
    code = models.CharField(max_length=10)    # 종목코드
    name = models.CharField(max_length=50)    # 종목명
    date = models.DateField()                 # 날짜
    close = models.FloatField()               # 종가
    high = models.FloatField()                # 고가
    low = models.FloatField()                 # 저가
    open = models.FloatField()                # 시가
    volume = models.BigIntegerField()         # 거래량
    change_rate = models.FloatField()         # 등락률(%)

    class Meta:
        unique_together = ('code', 'date')
        indexes = [
            models.Index(fields=['code', 'date']),
        ]

    def __str__(self):
        return f"{self.market} {self.code} {self.name} {self.date}"
