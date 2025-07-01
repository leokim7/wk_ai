// 전역 함수들
window.WalletKeeper = {
    // TX Hash 유효성 검증
    validateTxHash: function(txHash) {
        const regex = /^0x[a-fA-F0-9]{64}$/;
        return regex.test(txHash);
    },

    // 숫자 포맷팅
    formatNumber: function(num, decimals = 6) {
        return parseFloat(num).toFixed(decimals);
    },

    // 알림 표시
    showAlert: function(message, type = 'info') {
        const alertDiv = document.createElement('div');
        alertDiv.className = `alert alert-${type} alert-dismissible fade show`;
        alertDiv.innerHTML = `
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        `;
        
        const container = document.querySelector('.container');
        container.insertBefore(alertDiv, container.firstChild);
        
        setTimeout(() => {
            if (alertDiv.parentNode) {
                alertDiv.remove();
            }
        }, 5000);
    },

    // 클립보드 복사
    copyToClipboard: function(text) {
        navigator.clipboard.writeText(text).then(() => {
            this.showAlert('클립보드에 복사되었습니다!', 'success');
        }).catch(() => {
            this.showAlert('복사에 실패했습니다.', 'danger');
        });
    }
};

// DOM 로드 완료 시 초기화
document.addEventListener('DOMContentLoaded', function() {
    // 모든 폼에 로딩 표시 추가
    document.querySelectorAll('form').forEach(form => {
        form.addEventListener('submit', function() {
            const txHashInput = form.querySelector('input[name="tx_hash"]');
            if (txHashInput && !WalletKeeper.validateTxHash(txHashInput.value)) {
                return;
            }
        });
    });

    // TX Hash 입력 필드 실시간 검증
    document.querySelectorAll('input[name="tx_hash"]').forEach(input => {
        input.addEventListener('input', function() {
            const isValid = WalletKeeper.validateTxHash(this.value);
            
            this.classList.toggle('is-valid', isValid && this.value.length === 66);
            this.classList.toggle('is-invalid', this.value.length > 0 && !isValid);
            
            const submitBtn = this.closest('form').querySelector('button[type="submit"]');
            if (submitBtn) {
                submitBtn.disabled = !isValid && this.value.length > 0;
            }
        });
    });

    // 예시 버튼 클릭 이벤트
    document.querySelectorAll('.example-btn').forEach(btn => {
        btn.addEventListener('click', function() {
            const txHash = this.dataset.tx;
            const chainId = this.dataset.chain;
            
            const txInput = document.querySelector('input[name="tx_hash"]');
            const chainSelect = document.querySelector('select[name="chain_id"]');
            
            if (txInput) txInput.value = txHash;
            if (chainSelect) chainSelect.value = chainId;
            
            if (txInput) {
                txInput.dispatchEvent(new Event('input'));
            }
            
            txInput.scrollIntoView({ behavior: 'smooth', block: 'center' });
            
            setTimeout(() => {
                txInput.focus();
                txInput.select();
            }, 500);
        });
    });

    // 애니메이션 추가
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('fade-in');
            }
        });
    }, { threshold: 0.1 });

    document.querySelectorAll('.card').forEach(card => {
        observer.observe(card);
    });
});