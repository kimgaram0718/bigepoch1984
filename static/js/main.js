// 전역 변수
let predictionChart = null;

/*
// 샘플 데이터 (실제로는 API나 서버에서 받아와야 함)
const sampleData = {
    marketIndices: [
        {
            name: '코스피',
            date: '2024-03-19',
            closePrice: 2750.23,
            changeValue: 15.67,
            changePercent: 0.57
        },
        {
            name: '코스닥',
            date: '2024-03-19',
            closePrice: 850.45,
            changeValue: -5.32,
            changePercent: -0.62
        }
    ],
    topGainers: {
        kospi: [
            { name: '삼성전자', code: '005930', change: 2.5, close: 75000 },
            { name: 'SK하이닉스', code: '000660', change: 1.8, close: 145000 },
            { name: 'LG에너지솔루션', code: '373220', change: 1.5, close: 425000 },
            { name: '삼성바이오로직스', code: '207940', change: 1.2, close: 825000 },
            { name: 'NAVER', code: '035420', change: 1.0, close: 195000 }
        ],
        kosdaq: [
            { name: '셀트리온', code: '068270', change: 3.2, close: 185000 },
            { name: 'LG화학', code: '051910', change: 2.8, close: 425000 },
            { name: 'SK이노베이션', code: '096770', change: 2.5, close: 115000 },
            { name: '현대차', code: '005380', change: 2.2, close: 185000 },
            { name: '기아', code: '000270', change: 2.0, close: 115000 }
        ]
    }
};
*/

// DOM이 로드된 후 실행
$(document).ready(function() {
    // 초기 데이터 로드
    // loadMarketIndices();
    // loadTopGainers();
    // 이벤트 리스너 등록
    // $('#stockPredictionForm').on('submit', function(e) {
    //     e.preventDefault();
    //     handleStockSearch();
    // });
    $('#ajaxPredictButton').on('click', function() {
        const stockQuery = $('#stockQueryInput').val().trim();
        if (!stockQuery) {
            alert('종목명 또는 코드를 입력해주세요.');
            return;
        }
        $('#ajaxLoadingIndicator').show();
        $.post('/ajax_lstm_predict/', { stock_code: stockQuery }, function(data) {
            $('#ajaxLoadingIndicator').hide();
            if (!Array.isArray(data.future_preds) || data.future_preds.length === 0) {
                alert(data.error || "예측 결과 데이터가 없습니다.");
                return;
            }
            // 예측 결과 표
            let html = '<h4>실시간 3일 예측 결과</h4>';
            html += '<table class="table table-sm table-striped"><thead><tr><th>예측일</th><th>예측 종가</th></tr></thead><tbody>';
            const today = new Date();
            for (let i = 0; i < data.future_preds.length; i++) {
                const d = new Date(today.getTime() + (i+1)*24*60*60*1000);
                html += `<tr><td>${d.toISOString().slice(0,10)}</td><td>${Math.round(data.future_preds[i]).toLocaleString()} 원</td></tr>`;
            }
            html += '</tbody></table>';
            // MAE, 예측률, 정확도 표시
            let mae = data.mae ? data.mae.toFixed(2) : '-';
            let acc = data.accuracy !== undefined ? (data.accuracy * 100).toFixed(2) + '%' : '-';
            let r2 = data.r2 !== undefined ? (data.r2 * 100).toFixed(2) + '%' : '-';
            html += `<div>테스트 MAE(최근 3일): <b>${mae} 원</b> | 예측률(R²): <b>${r2}</b> | 정확도: <b>${acc}</b></div>`;
            $('#initialPredictionData').html(html);
            // 예측 결과 그래프
            const ctx = document.getElementById('predictionChart').getContext('2d');
            if (window.predictionChart) window.predictionChart.destroy();
            window.predictionChart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: [1,2,3].map(i => `+${i}일`),
                    datasets: [{
                        label: '예측 종가',
                        data: data.future_preds,
                        borderColor: '#28a745',
                        backgroundColor: 'rgba(40,167,69,0.1)',
                        borderWidth: 2,
                        fill: true,
                        tension: 0.4
                    }]
                },
                options: {
                    responsive: true,
                    plugins: {
                        title: { display: true, text: '실시간 3일 종가 예측 그래프' },
                        tooltip: {
                            callbacks: {
                                label: function(context) {
                                    return `예측 종가: ${context.raw.toLocaleString()}원`;
                                }
                            }
                        }
                    },
                    scales: {
                        y: {
                            beginAtZero: false,
                            ticks: {
                                callback: function(value) {
                                    return value.toLocaleString() + '원';
                                }
                            }
                        }
                    }
                }
            });
        });
    });
});

/*
// 시장 지수 정보 로드
function loadMarketIndices() {
    // ... 샘플 데이터 관련 코드 전체 주석처리 ...
}

// 급등주 정보 로드
function loadTopGainers() {
    // ... 샘플 데이터 관련 코드 전체 주석처리 ...
}
*/

// 종목 검색 처리
function handleStockSearch() {
    const stockQuery = $('#stockQueryInput').val().trim();
    if (!stockQuery) {
        alert('종목명 또는 코드를 입력해주세요.');
        return;
    }
    // 로딩 표시
    $('#ajaxLoadingIndicator').show();
    // 실제로는 API 호출이 들어가야 함
    setTimeout(() => {
        // 샘플 예측 데이터 (이 부분도 실제 API 연동 시 삭제)
        const predictions = [
            { date: '2024-03-20', price: 75200 },
            { date: '2024-03-21', price: 75800 },
            { date: '2024-03-22', price: 76200 },
            { date: '2024-03-25', price: 76800 },
            { date: '2024-03-26', price: 77500 },
            { date: '2024-03-27', price: 78200 },
            { date: '2024-03-28', price: 79000 }
        ];
        displayPredictions(stockQuery, predictions);
        $('#ajaxLoadingIndicator').hide();
    }, 1000);
}

// 예측 결과 표시
function displayPredictions(stockName, predictions) {
    // 제목 업데이트
    $('#stockTitle').text(`${stockName} - 7 영업일 예측 결과`);
    // 테이블 생성
    let tableHtml = `
        <table class="table table-sm table-striped table-hover prediction-table">
            <thead><tr><th>날짜</th><th>예측 종가</th></tr></thead>
            <tbody>
    `;
    predictions.forEach(pred => {
        tableHtml += `
            <tr>
                <td>${pred.date}</td>
                <td>${pred.price.toLocaleString()} 원</td>
            </tr>
        `;
    });
    tableHtml += '</tbody></table>';
    $('#initialPredictionData').html(tableHtml);
    // 차트 업데이트
    updatePredictionChart(predictions);
}

// 예측 차트 업데이트
function updatePredictionChart(predictions) {
    const ctx = document.getElementById('predictionChart').getContext('2d');
    // 기존 차트가 있다면 제거
    if (predictionChart) {
        predictionChart.destroy();
    }
    // 새 차트 생성
    predictionChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: predictions.map(p => p.date),
            datasets: [{
                label: '예측 종가',
                data: predictions.map(p => p.price),
                borderColor: '#007bff',
                backgroundColor: 'rgba(0, 123, 255, 0.1)',
                borderWidth: 2,
                fill: true,
                tension: 0.4
            }]
        },
        options: {
            responsive: true,
            plugins: {
                title: {
                    display: true,
                    text: '7일 예측 종가 추이'
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return `예측 종가: ${context.raw.toLocaleString()}원`;
                        }
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: false,
                    ticks: {
                        callback: function(value) {
                            return value.toLocaleString() + '원';
                        }
                    }
                }
            }
        }
    });
}

// 선택 종목 정보 보기 버튼 클릭 시 최근 1달간 종가 그래프로 제목 복구
$('#stockPredictionForm').on('submit', function(e) {
    e.preventDefault();
    // ... 기존 검색 처리 코드 ...
    // 차트 제목 복구
    if (window.predictionChart) {
        window.predictionChart.options.plugins.title.text = '최근 1달간 종가 그래프';
        window.predictionChart.update();
    }
}); 