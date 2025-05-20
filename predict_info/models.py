from django.db import models
from django.utils import timezone

class MarketIndex(models.Model):
    """
    일별 시장 지수(코스피, 코스닥) 정보를 저장하는 모델
    """
    market_choices = [
        ('KOSPI', '코스피'),
        ('KOSDAQ', '코스닥'),
    ]
    market_name = models.CharField(max_length=10, choices=market_choices, verbose_name="시장 구분")
    date = models.DateField(verbose_name="날짜")
    close_price = models.FloatField(verbose_name="종가")
    previous_day_close_price = models.FloatField(verbose_name="전일 종가", null=True, blank=True)
    change_value = models.FloatField(verbose_name="대비", null=True, blank=True)
    change_percent = models.FloatField(verbose_name="등락률 (%)", null=True, blank=True)
    volume = models.BigIntegerField(verbose_name="거래량", null=True, blank=True)
    trade_value = models.BigIntegerField(verbose_name="거래대금", null=True, blank=True)

    class Meta:
        verbose_name = "일별 시장 지수"
        verbose_name_plural = "일별 시장 지수 목록"
        unique_together = ('market_name', 'date')
        ordering = ['-date', 'market_name']

    def __str__(self):
        return f"{self.date} - {self.get_market_name_display()}: {self.close_price}"

    def calculate_changes(self):
        """등락 및 등락률을 계산하여 필드에 설정합니다."""
        if self.previous_day_close_price is not None and self.close_price is not None:
            self.change_value = self.close_price - self.previous_day_close_price
            if self.previous_day_close_price != 0:
                self.change_percent = ((self.close_price - self.previous_day_close_price) / self.previous_day_close_price) * 100
            else:
                # 전일 종가가 0이고 현재 종가도 0이면 등락률 0, 아니면 무한대 또는 특정 값
                self.change_percent = 0 if self.close_price == self.previous_day_close_price else float('inf')
        else:
            self.change_value = None
            self.change_percent = None

    def save(self, *args, **kwargs):
        self.calculate_changes() # 저장 전에 항상 등락률 계산
        super().save(*args, **kwargs)


class StockPrice(models.Model):
    """
    개별 종목의 일별 가격 정보를 저장하는 모델
    """
    stock_code = models.CharField(max_length=20, verbose_name="종목 코드")
    stock_name = models.CharField(max_length=100, verbose_name="종목명")
    market_name = models.CharField(max_length=10, verbose_name="시장 구분", help_text="예: KOSPI, KOSDAQ")
    date = models.DateField(verbose_name="날짜")
    open_price = models.FloatField(verbose_name="시가", null=True, blank=True)
    high_price = models.FloatField(verbose_name="고가", null=True, blank=True)
    low_price = models.FloatField(verbose_name="저가", null=True, blank=True)
    close_price = models.FloatField(verbose_name="종가")
    previous_day_close_price = models.FloatField(verbose_name="전일 종가", null=True, blank=True)
    change_value = models.FloatField(verbose_name="대비", null=True, blank=True)
    change_percent = models.FloatField(verbose_name="등락률 (%)", null=True, blank=True)
    volume = models.BigIntegerField(verbose_name="거래량", null=True, blank=True)
    trade_value = models.BigIntegerField(verbose_name="거래대금", null=True, blank=True)

    class Meta:
        verbose_name = "개별 종목 가격"
        verbose_name_plural = "개별 종목 가격 목록"
        unique_together = ('stock_code', 'date')
        ordering = ['-date', 'stock_name']
        indexes = [
            models.Index(fields=['date', 'market_name', '-change_percent']),
            models.Index(fields=['date', 'stock_name']),
            models.Index(fields=['date', 'stock_code']),
        ]

    def __str__(self):
        return f"{self.date} - {self.stock_name} ({self.stock_code}): {self.close_price}"

    def calculate_changes(self):
        """등락 및 등락률을 계산하여 필드에 설정합니다."""
        if self.previous_day_close_price is not None and self.close_price is not None:
            self.change_value = self.close_price - self.previous_day_close_price
            if self.previous_day_close_price != 0:
                self.change_percent = ((self.close_price - self.previous_day_close_price) / self.previous_day_close_price) * 100
            else:
                self.change_percent = 0 if self.close_price == self.previous_day_close_price else float('inf')
        else:
            self.change_value = None
            self.change_percent = None

    def save(self, *args, **kwargs):
        self.calculate_changes() # 저장 전에 항상 등락률 계산
        super().save(*args, **kwargs)

class PredictedStockPrice(models.Model):
    """
    일별로 예측된 개별 종목의 미래 가격 정보를 저장하는 모델
    """
    ANALYSIS_CHOICES = [
        ('technical', '기술적 분석'),
        ('comprehensive', '종합 분석'), # 종합 분석 모델이 있다면 추가
    ]
    stock_code = models.CharField(max_length=20, verbose_name="종목 코드", db_index=True)
    stock_name = models.CharField(max_length=100, verbose_name="종목명", blank=True) # 편의를 위해 추가
    market_name = models.CharField(max_length=10, verbose_name="시장 구분", blank=True) # 편의를 위해 추가

    prediction_base_date = models.DateField(verbose_name="예측 기준일", db_index=True, help_text="이 날짜의 종가까지를 기반으로 예측이 수행됨")
    predicted_date = models.DateField(verbose_name="예측 대상일", help_text="실제 주가가 예측된 미래의 날짜")
    predicted_price = models.FloatField(verbose_name="예측 종가")
    
    analysis_type = models.CharField(
        max_length=20, 
        choices=ANALYSIS_CHOICES, 
        default='technical', 
        verbose_name="분석 유형"
    )
    
    created_at = models.DateTimeField(auto_now_add=True, verbose_name="생성 시각")
    updated_at = models.DateTimeField(auto_now=True, verbose_name="수정 시각")

    class Meta:
        verbose_name = "종목별 예측 가격"
        verbose_name_plural = "종목별 예측 가격 목록"
        # 특정 종목, 특정 기준일, 특정 예측 대상일, 특정 분석 유형에 대한 예측은 유일해야 함
        unique_together = ('stock_code', 'prediction_base_date', 'predicted_date', 'analysis_type')
        ordering = ['-prediction_base_date', 'stock_code', 'predicted_date']
        indexes = [
            models.Index(fields=['stock_code', 'prediction_base_date', 'analysis_type', 'predicted_date'], name='pred_price_lookup_idx'),
            models.Index(fields=['prediction_base_date']), # 오래된 예측 삭제 시 사용
        ]

    def __str__(self):
        return f"{self.stock_name}({self.stock_code}) - 기준일:{self.prediction_base_date} -> 예측일:{self.predicted_date} = {self.predicted_price} ({self.get_analysis_type_display()})"

