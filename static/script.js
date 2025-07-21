/**
 * 疾病预测与数据分析系统 - 前端性能优化版本
 * 包含了高效动画、懒加载和页面性能优化
 */

// 配置参数
const CONFIG = {
  // 基本配置
  SCROLL_THRESHOLD: 200,        // 滚动触发阈值
  SCROLL_DEBOUNCE: 20,          // 滚动防抖时间（毫秒）
  ANIMATION_THRESHOLD: 0.2,     // 动画触发阈值（元素在视口中的比例）
  IS_MOBILE: window.innerWidth < 768, // 是否为移动设备
  
  // 性能设置
  ENABLE_ANIMATIONS: true,      // 全局动画开关
  LAZY_LOAD_IMAGES: true,       // 图片懒加载
  ENABLE_TRANSITIONS: true,     // 启用页面过渡
  
  // CSS类名
  CLASSES: {
    visible: 'is-visible',
    fadeInSection: 'fade-in-section',
    scrollToTop: 'scroll-to-top',
    scrollIndicator: 'scroll-indicator'
  }
};

/**
 * 工具函数
 */
const Utils = {
  // 防抖函数 - 减少高频事件触发次数
  debounce(func, wait) {
    let timeout;
    return function(...args) {
      clearTimeout(timeout);
      timeout = setTimeout(() => func.apply(this, args), wait);
    };
  },
  
  // 检测元素是否在视口内
  isElementInViewport(el, threshold = CONFIG.ANIMATION_THRESHOLD) {
    if (!el) return false;
    
    const rect = el.getBoundingClientRect();
    const windowHeight = window.innerHeight || document.documentElement.clientHeight;
    
    // 元素顶部进入视口底部一定比例后视为可见
    return (
      rect.top <= windowHeight * (1 - threshold) &&
      rect.bottom >= 0
    );
  },
  
  // 检测设备性能
  checkDevicePerformance() {
    // 检测移动设备
    if (/Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent)) {
      CONFIG.IS_MOBILE = true;
    }
    
    // 如果设备是低端移动设备，可以禁用一些复杂动画
    if (CONFIG.IS_MOBILE && navigator.hardwareConcurrency && navigator.hardwareConcurrency < 4) {
      CONFIG.ENABLE_ANIMATIONS = false;
    }
  },
  
  // 添加一次性事件监听
  once(element, eventType, callback) {
    const handler = (e) => {
      callback(e);
      element.removeEventListener(eventType, handler);
    };
    element.addEventListener(eventType, handler);
  }
};

/**
 * 动画控制器
 */
const AnimationController = {
  // 初始化可见性观察器
  initIntersectionObserver() {
    if (!('IntersectionObserver' in window) || !CONFIG.ENABLE_ANIMATIONS) {
      // 如果不支持IntersectionObserver或禁用动画，则使所有元素立即可见
      document.querySelectorAll(`.${CONFIG.CLASSES.fadeInSection}`).forEach(el => {
        el.classList.add(CONFIG.CLASSES.visible);
      });
      return;
    }
    
    const options = {
      root: null,
      rootMargin: '0px',
      threshold: 0.15
    };
    
    const observer = new IntersectionObserver((entries) => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          entry.target.classList.add(CONFIG.CLASSES.visible);
          observer.unobserve(entry.target); // 元素显示后不再观察
        }
      });
    }, options);
    
    // 观察所有需要动画的元素
    document.querySelectorAll(`.${CONFIG.CLASSES.fadeInSection}`).forEach(el => {
      observer.observe(el);
    });
  },
  
  // 准备元素动画
  prepareElements() {
    // 将所有内容部分转换为可动画元素
    const sections = document.querySelectorAll('.jumbotron, .row, .card-container, .chart-container, .content-block, .slide-section');
    
    sections.forEach(section => {
      if (!section.classList.contains(CONFIG.CLASSES.fadeInSection)) {
        section.classList.add(CONFIG.CLASSES.fadeInSection);
      }
    });
    
    // 如果禁用动画，则立即将所有元素显示出来
    if (!CONFIG.ENABLE_ANIMATIONS) {
      sections.forEach(section => {
        section.classList.add(CONFIG.CLASSES.visible);
      });
    }
  }
};

/**
 * 页面滚动控制
 */
const ScrollController = {
  // 滚动指示器
  initScrollIndicator() {
    const scrollIndicator = document.querySelector(`.${CONFIG.CLASSES.scrollIndicator}`) || 
                           this.createScrollIndicator();
    
    this.updateScrollIndicator = () => {
      const windowHeight = document.documentElement.scrollHeight - document.documentElement.clientHeight;
      const scrolled = (window.scrollY / windowHeight) * 100;
      
      if (scrollIndicator) {
        scrollIndicator.style.width = scrolled + '%';
      }
    };
  },
  
  // 创建滚动指示器元素
  createScrollIndicator() {
    const indicator = document.createElement('div');
    indicator.className = CONFIG.CLASSES.scrollIndicator;
    document.body.appendChild(indicator);
    return indicator;
  },
  
  // 初始化回到顶部按钮
  initScrollToTopButton() {
    const scrollToTopBtn = document.querySelector(`.${CONFIG.CLASSES.scrollToTop}`) || 
                          this.createScrollToTopButton();
    
    // 更新按钮可见性
    this.updateScrollToTopButton = () => {
      if (window.scrollY > CONFIG.SCROLL_THRESHOLD) {
        scrollToTopBtn.classList.add('visible');
      } else {
        scrollToTopBtn.classList.remove('visible');
      }
    };
  },
  
  // 创建回到顶部按钮
  createScrollToTopButton() {
    const button = document.createElement('div');
    button.className = CONFIG.CLASSES.scrollToTop;
    button.innerHTML = '<i class="fas fa-arrow-up"></i>';
    button.addEventListener('click', () => {
      window.scrollTo({ top: 0, behavior: 'smooth' });
    });
    document.body.appendChild(button);
    return button;
  },
  
  // 处理滚动事件
  handleScroll() {
    // 更新滚动指示器
    this.updateScrollIndicator();
    
    // 更新回到顶部按钮
    this.updateScrollToTopButton();
  },
  
  // 初始化页面滚动处理
  init() {
    this.initScrollIndicator();
    this.initScrollToTopButton();
    
    // 使用防抖来优化滚动事件处理
    const debouncedScrollHandler = Utils.debounce(
      () => this.handleScroll(), 
      CONFIG.SCROLL_DEBOUNCE
    );
    
    window.addEventListener('scroll', debouncedScrollHandler, { passive: true });
    
    // 初始触发一次滚动处理
    this.handleScroll();
  }
};

/**
 * 图片懒加载控制器
 */
const LazyLoadController = {
  init() {
    if (!CONFIG.LAZY_LOAD_IMAGES) {
      // 如果禁用了懒加载，直接加载所有图片
      document.querySelectorAll('img[data-src]').forEach(img => {
        img.src = img.getAttribute('data-src');
        img.classList.add('loaded');
      });
      return;
    }
    
    if ('IntersectionObserver' in window) {
      const imgObserver = new IntersectionObserver((entries, observer) => {
        entries.forEach(entry => {
          if (entry.isIntersecting) {
            const img = entry.target;
            const src = img.getAttribute('data-src');
            
            if (src) {
              // 创建一个临时图像来预加载
              const tempImage = new Image();
              tempImage.onload = () => {
                img.src = src;
                img.classList.add('loaded');
              };
              tempImage.src = src;
              
              // 移除data-src属性并停止观察
              img.removeAttribute('data-src');
              observer.unobserve(img);
            }
          }
        });
      }, {
        rootMargin: '50px'  // 提前50px开始加载
      });

      // 观察所有懒加载图片
      document.querySelectorAll('img[data-src]').forEach(img => {
        img.classList.add('lazy');
        imgObserver.observe(img);
      });
    } else {
      // 回退方案：不支持IntersectionObserver时
      document.querySelectorAll('img[data-src]').forEach(img => {
        img.src = img.getAttribute('data-src');
        img.classList.add('loaded');
      });
    }
  }
};

/**
 * 页面转换控制器
 */
const PageTransitionController = {
  init() {
    if (!CONFIG.ENABLE_TRANSITIONS) return;
    
    // 创建页面过渡元素
    const transitionElement = document.createElement('div');
    transitionElement.className = 'page-transition';
    document.body.appendChild(transitionElement);
    
    // 处理站内链接点击
    document.querySelectorAll('a').forEach(link => {
      // 只处理同站链接，排除下拉菜单项
      if (link.hostname === window.location.hostname && 
          !link.hasAttribute('data-no-transition') && 
          !link.classList.contains('dropdown-item')) {
        
        link.addEventListener('click', (e) => {
          // 如果是同一页面内的锚点链接，使用平滑滚动
          if (link.hash && document.querySelector(link.hash)) {
            e.preventDefault();
            const targetElement = document.querySelector(link.hash);
            
            // 平滑滚动到目标元素
            window.scrollTo({
              top: targetElement.offsetTop - 70,
              behavior: 'smooth'
            });
          } 
          // 如果是站内其他页面链接，添加过渡效果
          else if (!link.hash && link.pathname !== window.location.pathname) {
            e.preventDefault();
            
            // 激活过渡效果
            transitionElement.classList.add('active');
            
            // 300ms后跳转到目标页面
            setTimeout(() => {
              window.location.href = link.href;
            }, 300);
          }
        });
      }
    });
    
    // 页面加载完成后的淡入效果
    document.body.classList.add('fade-in');
  }
};

/**
 * 标签页控制器
 */
const TabController = {
  init() {
    const tabLinks = document.querySelectorAll('[data-bs-toggle="tab"]');
    
    tabLinks.forEach(tabLink => {
      tabLink.addEventListener('shown.bs.tab', (e) => {
        const targetTabPane = document.querySelector(e.target.getAttribute('data-bs-target'));
        
        if (targetTabPane) {
          // 确保内容进入视口
          setTimeout(() => {
            // 使用更高效的scrollIntoView方式
            targetTabPane.scrollIntoView({
              behavior: 'smooth',
              block: 'nearest'
            });
          }, 50);
          
          // 激活当前标签页中的图表和图片
          this.activateTabContent(targetTabPane);
        }
      });
    });
  },
  
  // 激活标签页中的内容
  activateTabContent(container) {
    // 加载懒加载图片
    container.querySelectorAll('img[data-src]').forEach(img => {
      if (img.getAttribute('data-src')) {
        img.src = img.getAttribute('data-src');
        img.classList.add('loaded');
        img.removeAttribute('data-src');
      }
    });
    
    // 让所有动画元素可见
    container.querySelectorAll(`.${CONFIG.CLASSES.fadeInSection}`).forEach(el => {
      el.classList.add(CONFIG.CLASSES.visible);
    });
  }
};

/**
 * 背景控制器
 */
const BackgroundController = {
  init() {
    const backgrounds = document.querySelectorAll('.parallax-bg');
    if (backgrounds.length === 0) return;
    
    let currentBg = 0;
    backgrounds[0].classList.add('active');
    
    // 使用防抖优化性能
    const debouncedBackgroundHandler = Utils.debounce(() => {
      const scrollPosition = window.scrollY;
      const windowHeight = window.innerHeight;
      const documentHeight = document.body.scrollHeight;
      
      // 根据滚动位置计算应该显示哪张背景图
      const scrollRatio = Math.min(scrollPosition / (documentHeight - windowHeight), 1);
      const bgIndex = Math.min(Math.floor(scrollRatio * backgrounds.length), backgrounds.length - 1);
      
      // 如果背景图需要更换
      if (bgIndex !== currentBg) {
        // 移除当前激活的背景图
        backgrounds[currentBg].classList.remove('active');
        // 激活新的背景图
        backgrounds[bgIndex].classList.add('active');
        // 更新当前背景图索引
        currentBg = bgIndex;
      }
    }, 100);
    
    window.addEventListener('scroll', debouncedBackgroundHandler, { passive: true });
  }
};

/**
 * 预测表单控制器
 */
const PredictionFormController = {
  init() {
    // 检查是否在预测页面
    const predictionForm = document.querySelector('#prediction-form');
    if (!predictionForm) return;
    
    // 为预测表单添加提交事件监听
    predictionForm.addEventListener('submit', (e) => {
      e.preventDefault();
      
      // 显示加载动画
      const submitBtn = predictionForm.querySelector('button[type="submit"]');
      const originalBtnText = submitBtn.innerHTML;
      submitBtn.disabled = true;
      submitBtn.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> 预测中...';
      
      // 获取表单数据
      const formData = new FormData(predictionForm);
      
      // 发送AJAX请求
      fetch('/predict', {
        method: 'POST',
        body: formData
      })
      .then(response => {
        if (!response.ok) {
          throw new Error('服务器响应错误：' + response.status);
        }
        return response.json();
      })
      .then(data => {
        // 恢复提交按钮
        submitBtn.disabled = false;
        submitBtn.innerHTML = originalBtnText;
        
        // 显示预测结果
        if (data.error) {
          this.showPredictionError(new Error(data.error));
        } else {
          this.showPredictionResult(data);
        }
      })
      .catch(error => {
        // 恢复提交按钮
        submitBtn.disabled = false;
        submitBtn.innerHTML = originalBtnText;
        
        // 显示错误
        this.showPredictionError(error);
      });
    });
    
    // 添加疾病类型切换事件
    const diseaseTypeSelect = document.querySelector('#disease_type');
    if (diseaseTypeSelect) {
      diseaseTypeSelect.addEventListener('change', () => {
        // 显示相应的表单字段
        this.toggleFormFields(diseaseTypeSelect.value);
      });
      
      // 初始化表单字段显示
      this.toggleFormFields(diseaseTypeSelect.value);
    }
  },
  
  // 切换表单字段
  toggleFormFields(diseaseType) {
    // 获取所有表单字段组
    const fieldGroups = document.querySelectorAll('.form-field-group');
    
    // 隐藏所有字段组
    fieldGroups.forEach(group => {
      group.style.display = 'none';
    });
    
    // 显示对应疾病类型的字段组
    const targetGroup = document.querySelector(`.form-field-group[data-disease="${diseaseType}"]`);
    if (targetGroup) {
      targetGroup.style.display = 'block';
      
      // 添加淡入效果
      setTimeout(() => {
        targetGroup.classList.add('fade-in');
      }, 50);
    }
  },
  
  // 显示预测结果
  showPredictionResult(data) {
    // 获取结果容器
    const resultContainer = document.querySelector('#prediction-result');
    if (!resultContainer) return;
    
    // 清空现有内容
    resultContainer.innerHTML = '';
    resultContainer.style.display = 'block';
    
    // 创建结果卡片
    const resultCard = document.createElement('div');
    resultCard.className = 'card fade-in';
    
    // 根据疾病类型创建相应的结果卡片内容
    if (data.disease_type === 'stroke') {
      this.createStrokeResultCard(resultCard, data);
    } else if (data.disease_type === 'heart') {
      this.createHeartResultCard(resultCard, data);
    } else if (data.disease_type === 'cirrhosis') {
      this.createCirrhosisResultCard(resultCard, data);
    }
    
    // 如果使用了模拟数据，添加提示信息
    if (data.note && data.note.includes('模拟数据')) {
      const noteAlert = document.createElement('div');
      noteAlert.className = 'alert alert-info mt-3';
      noteAlert.innerHTML = `
        <small><i class="fas fa-info-circle"></i> 注意：${data.note}。实际结果可能有所不同。</small>
      `;
      resultCard.appendChild(noteAlert);
    }
    
    // 添加到结果容器
    resultContainer.appendChild(resultCard);
    
    // 滚动到结果区域
    resultContainer.scrollIntoView({behavior: 'smooth', block: 'nearest'});
    
    // 初始化风险仪表盘
    this.initRiskGauge();
  },
  
  // 创建中风结果卡片
  createStrokeResultCard(card, data) {
    const isPredicted = data.prediction === 1;
    const probability = data.probability["1"] * 100;
    
    // 检查是否有原始概率数据（校准前）
    let calibrationInfo = '';
    if (data.raw_probability && data.calibrated) {
      const rawProbability = data.raw_probability["1"] * 100;
      calibrationInfo = `
        <div class="alert alert-info mt-3">
          <h5><i class="fas fa-info-circle"></i> 模型校准信息</h5>
          <p>原始预测风险: <strong>${rawProbability.toFixed(2)}%</strong> → 校准后风险: <strong>${probability.toFixed(2)}%</strong></p>
          <small>我们应用了先进的概率校准技术，特别针对高风险人群优化了预测结果，使其更加符合实际临床情况。</small>
        </div>
      `;
    }
    
    card.innerHTML = `
      <div class="card-header bg-${isPredicted ? 'danger' : 'success'} text-white">
        <h3 class="card-title mb-0">中风风险预测结果</h3>
      </div>
      <div class="card-body">
        <div class="text-center mb-4">
          <div class="risk-gauge" data-risk="${probability}">
            <div class="gauge-value">${probability.toFixed(2)}%</div>
          </div>
        </div>
        <h4 class="text-center mb-3">风险评估: <span class="${isPredicted ? 'text-danger' : 'text-success'}">${isPredicted ? '高风险' : '低风险'}</span></h4>
        <div class="progress mb-4">
          <div class="progress-bar bg-${this.getProgressBarColor(probability)}" role="progressbar" style="width: ${probability}%" aria-valuenow="${probability}" aria-valuemin="0" aria-valuemax="100"></div>
        </div>
        ${calibrationInfo}
        <div class="alert alert-${isPredicted ? 'warning' : 'info'}">
          <p>${isPredicted ? 
            '根据您提供的数据，模型预测您有较高的中风风险。请考虑尽快咨询医生进行更详细的评估。' : 
            '根据您提供的数据，模型预测您目前的中风风险较低。继续保持健康的生活方式！'}
          </p>
        </div>
        <div class="mt-4">
          <h5>预防建议:</h5>
          <ul>
            <li>定期监测血压，保持在健康范围内</li>
            <li>保持健康饮食，减少钠和饱和脂肪的摄入</li>
            <li>每周进行至少150分钟的中等强度有氧运动</li>
            <li>戒烟限酒</li>
            <li>定期体检，特别关注心血管系统健康</li>
          </ul>
        </div>
      </div>
    `;
  },
  
  // 创建心脏病结果卡片
  createHeartResultCard(card, data) {
    const isPredicted = data.prediction === 1;
    const probability = data.probability["1"] * 100;
    
    // 检查是否有原始概率数据（校准前）
    let calibrationInfo = '';
    if (data.raw_probability && data.calibrated) {
      const rawProbability = data.raw_probability["1"] * 100;
      calibrationInfo = `
        <div class="alert alert-info mt-3">
          <h5><i class="fas fa-info-circle"></i> 模型校准信息</h5>
          <p>原始预测风险: <strong>${rawProbability.toFixed(2)}%</strong> → 校准后风险: <strong>${probability.toFixed(2)}%</strong></p>
          <small>我们应用了先进的概率校准技术，特别针对高风险人群优化了预测结果，使其更加符合实际临床情况。</small>
        </div>
      `;
    }
    
    card.innerHTML = `
      <div class="card-header bg-${isPredicted ? 'danger' : 'success'} text-white">
        <h3 class="card-title mb-0">心脏病风险预测结果</h3>
      </div>
      <div class="card-body">
        <div class="text-center mb-4">
          <div class="risk-gauge" data-risk="${probability}">
            <div class="gauge-value">${probability.toFixed(2)}%</div>
          </div>
        </div>
        <h4 class="text-center mb-3">风险评估: <span class="${isPredicted ? 'text-danger' : 'text-success'}">${isPredicted ? '高风险' : '低风险'}</span></h4>
        <div class="progress mb-4">
          <div class="progress-bar bg-${this.getProgressBarColor(probability)}" role="progressbar" style="width: ${probability}%" aria-valuenow="${probability}" aria-valuemin="0" aria-valuemax="100"></div>
        </div>
        ${calibrationInfo}
        <div class="alert alert-${isPredicted ? 'warning' : 'info'}">
          <p>${isPredicted ? 
            '根据您提供的数据，模型预测您有较高的心脏病风险。建议您咨询心脏科医生进行进一步评估。' : 
            '根据您提供的数据，模型预测您目前的心脏病风险较低。请继续保持健康的生活方式！'}
          </p>
        </div>
        <div class="mt-4">
          <h5>预防建议:</h5>
          <ul>
            <li>控制血压和胆固醇水平</li>
            <li>每天进行适量运动</li>
            <li>均衡饮食，增加蔬果摄入</li>
            <li>保持健康体重</li>
            <li>管理压力，保证充足睡眠</li>
          </ul>
        </div>
      </div>
    `;
  },
  
  // 创建肝硬化结果卡片
  createCirrhosisResultCard(card, data) {
    const stagePrediction = Math.round(data.prediction);
    const stageDescriptions = {
      1: '早期肝硬化，肝脏仍能正常运作',
      2: '中度肝硬化，肝脏功能轻度受损',
      3: '进展期肝硬化，肝脏功能中度受损',
      4: '晚期肝硬化，肝脏功能严重受损'
    };
    
    const stageRiskClass = {
      1: 'success',
      2: 'info',
      3: 'warning',
      4: 'danger'
    };
    
    // 检查是否有原始预测数据（校准前）
    let calibrationInfo = '';
    if (data.raw_prediction && data.calibrated) {
      const rawStagePrediction = Math.round(data.raw_prediction);
      calibrationInfo = `
        <div class="alert alert-info mt-3">
          <h5><i class="fas fa-info-circle"></i> 模型校准信息</h5>
          <p>原始预测分期: <strong>${rawStagePrediction} 期</strong> → 校准后分期: <strong>${stagePrediction} 期</strong></p>
          <small>我们应用了先进的概率校准技术，特别针对高风险人群优化了预测结果，使其更加符合实际临床情况。</small>
        </div>
      `;
    }
    
    card.innerHTML = `
      <div class="card-header bg-${stageRiskClass[stagePrediction]} text-${stagePrediction === 4 ? 'white' : 'dark'}">
        <h3 class="card-title mb-0">肝硬化分期预测结果</h3>
      </div>
      <div class="card-body">
        <div class="text-center mb-4">
          <div class="stage-indicator">
            <span class="stage-number stage-${stagePrediction}">${stagePrediction}</span>
            <span class="stage-label">期</span>
          </div>
        </div>
        <h4 class="text-center mb-3">阶段评估: <span class="text-${stageRiskClass[stagePrediction]}">${stageDescriptions[stagePrediction]}</span></h4>
        <div class="progress mb-4">
          <div class="progress-bar bg-${stageRiskClass[stagePrediction]}" role="progressbar" style="width: ${stagePrediction * 25}%" aria-valuenow="${stagePrediction * 25}" aria-valuemin="0" aria-valuemax="100"></div>
        </div>
        ${calibrationInfo}
        <div class="alert alert-${stageRiskClass[stagePrediction]}">
          <p>根据您提供的数据，模型预测您的肝硬化处于第 ${stagePrediction} 期。${
            stagePrediction <= 2 ? 
            '早期发现很重要，建议定期随访并遵循医嘱。' : 
            '请务必遵循专业医生的治疗建议，并定期检查肝功能。'
          }</p>
        </div>
        <div class="mt-4">
          <h5>健康管理建议:</h5>
          <ul>
            <li>严格遵医嘱用药</li>
            <li>保持健康饮食，限制盐分摄入</li>
            <li>避免饮酒和其他肝毒性物质</li>
            <li>定期监测肝功能</li>
            <li>适当休息，避免过度劳累</li>
          </ul>
        </div>
      </div>
    `;
  },
  
  // 显示预测错误
  showPredictionError(error) {
    const resultContainer = document.querySelector('#prediction-result');
    if (!resultContainer) return;
    
    resultContainer.innerHTML = '';
    resultContainer.style.display = 'block';
    
    const errorAlert = document.createElement('div');
    errorAlert.className = 'alert alert-danger fade-in';
    
    // 检查是否是模型未加载错误
    const isModelError = error.message && 
      (error.message.includes('模型未加载') || 
       error.message.includes('model') || 
       error.message.includes('加载失败'));
    
    if (isModelError) {
      errorAlert.innerHTML = `
        <h4 class="alert-heading"><i class="fas fa-exclamation-triangle"></i> 出错了</h4>
        <p>很抱歉，预测过程中遇到问题：模型未加载，请稍后再试</p>
        <hr>
        <p class="mb-0">请检查您的输入数据并重试。</p>
      `;
    } else {
      errorAlert.innerHTML = `
        <h4 class="alert-heading"><i class="fas fa-exclamation-triangle"></i> 预测出错</h4>
        <p>很抱歉，预测过程中出现了错误。请检查输入数据并重试。</p>
        <hr>
        <p class="mb-0">错误详情: ${error.message || '未知错误'}</p>
      `;
    }
    
    resultContainer.appendChild(errorAlert);
    resultContainer.scrollIntoView({behavior: 'smooth', block: 'nearest'});
  },
  
  // 初始化风险仪表盘
  initRiskGauge() {
    const gauges = document.querySelectorAll('.risk-gauge');
    
    gauges.forEach(gauge => {
      const risk = parseFloat(gauge.dataset.risk);
      const angle = (risk / 100) * 180 - 90; // 转换为角度，-90到90度
      
      // 设置仪表盘样式 - 使用CSS变量避免重绘
      gauge.style.setProperty('--risk-angle', `${angle}deg`);
      gauge.style.background = `conic-gradient(
        ${this.getRiskColor(risk)} ${angle}deg,
        #e0e0e0 ${angle}deg
      )`;
    });
  },
  
  // 根据风险值获取进度条颜色
  getProgressBarColor(riskValue) {
    if (riskValue < 20) return 'success';
    if (riskValue < 50) return 'info';
    if (riskValue < 80) return 'warning';
    return 'danger';
  },
  
  // 根据风险值获取颜色
  getRiskColor(risk) {
    if (risk < 20) return '#28a745'; // 绿色
    if (risk < 50) return '#17a2b8'; // 蓝色
    if (risk < 80) return '#ffc107'; // 黄色
    return '#dc3545'; // 红色
  }
};

/**
 * 导航菜单控制器
 */
const NavController = {
  init() {
    // 初始化移动端导航
    this.setupMobileNav();
  },
  
  setupMobileNav() {
    const navbarToggler = document.querySelector('.navbar-toggler');
    const navbarCollapse = document.querySelector('.navbar-collapse');
    
    if (navbarToggler && navbarCollapse) {
      navbarToggler.addEventListener('click', () => {
        navbarCollapse.classList.toggle('show');
      });
      
      // 点击导航项后自动关闭移动导航菜单
      const navLinks = navbarCollapse.querySelectorAll('.nav-link');
      navLinks.forEach(link => {
        link.addEventListener('click', () => {
          navbarCollapse.classList.remove('show');
        });
      });
    }
  }
};

/**
 * 主应用初始化
 */
document.addEventListener('DOMContentLoaded', () => {
  try {
    // 检测设备性能，决定是否启用动画
    Utils.checkDevicePerformance();
    
    // 初始化工具提示和弹出框
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    if(tooltipTriggerList.length > 0) {
        tooltipTriggerList.map(function (tooltipTriggerEl) {
            return new bootstrap.Tooltip(tooltipTriggerEl);
        });
    }
    
    // 初始化背景控制器
    if(typeof BackgroundController !== 'undefined') {
      BackgroundController.init();
    }
    
    // 准备动画元素
    if(typeof AnimationController !== 'undefined') {
      AnimationController.prepareElements();
      AnimationController.initIntersectionObserver();
    }
    
    // 初始化滚动控制
    if(typeof ScrollController !== 'undefined') {
      ScrollController.init();
    }
    
    // 初始化图片懒加载
    if(typeof LazyLoadController !== 'undefined') {
      LazyLoadController.init();
    }
    
    // 初始化页面过渡效果
    if(typeof PageTransitionController !== 'undefined') {
      PageTransitionController.init();
    }
    
    // 初始化标签页控制
    if(typeof TabController !== 'undefined') {
      TabController.init();
    }
    
    // 初始化预测表单控制
    if(typeof PredictionFormController !== 'undefined') {
      PredictionFormController.init();
    }
    
    // 初始化导航控制器
    if(typeof NavController !== 'undefined') {
      NavController.init();
    }
    
    // 初始化图表控制器
    if(typeof ChartController !== 'undefined') {
      ChartController.init();
    }
    
    // 先触发一次滚动事件，确保初始加载时的动画正常显示
    window.dispatchEvent(new Event('scroll'));
    
    console.log('所有系统组件已初始化');
  } catch(e) {
    console.error('初始化失败: ', e);
  }
}); 