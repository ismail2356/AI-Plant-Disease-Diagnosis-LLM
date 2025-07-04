// Botanical Tech Animated Background
class BotanicalTechBackground {
    constructor() {
        this.canvas = document.getElementById('background-canvas');
        this.ctx = this.canvas.getContext('2d');
        this.elements = [];
        
        this.mouseX = 0;
        this.mouseY = 0;
        this.lastMouseX = 0;
        this.lastMouseY = 0;
        this.mouseSpeed = 0;
        this.isMouseMoving = false;
        this.mouseTimer = null;
        
        this.resizeCanvas();
        this.createElements();
        
        // Mouse interactions
        window.addEventListener('mousemove', (e) => {
            this.mouseX = e.clientX;
            this.mouseY = e.clientY;
            
            // Calculate mouse speed
            const dx = this.mouseX - this.lastMouseX;
            const dy = this.mouseY - this.lastMouseY;
            this.mouseSpeed = Math.sqrt(dx * dx + dy * dy) * 0.1;
            if (this.mouseSpeed > 5) this.mouseSpeed = 5;
            
            this.lastMouseX = this.mouseX;
            this.lastMouseY = this.mouseY;
            
            this.isMouseMoving = true;
            
            // Reset mouse moving flag after 100ms of inactivity
            clearTimeout(this.mouseTimer);
            this.mouseTimer = setTimeout(() => {
                this.isMouseMoving = false;
                this.mouseSpeed = 0;
            }, 100);
        });
        
        window.addEventListener('resize', () => this.resizeCanvas());
        
        this.animate();
    }
    
    resizeCanvas() {
        this.canvas.width = window.innerWidth;
        this.canvas.height = window.innerHeight;
        
        // Resize olduğunda elementleri tekrar oluştur
        if (this.elements.length > 0) {
            this.elements = [];
            this.createElements();
        }
    }
    
    createElements() {
        // DNA sarmalları, yaprak desenleri, tohumlar, hücreler
        // DNA sarmalları
        for (let i = 0; i < 7; i++) {
            this.elements.push(new DNAHelix(
                Math.random() * this.canvas.width,
                Math.random() * this.canvas.height,
                this.ctx
            ));
        }
        
        // Yaprak desenleri
        for (let i = 0; i < 10; i++) {
            this.elements.push(new LeafPattern(
                Math.random() * this.canvas.width,
                Math.random() * this.canvas.height,
                this.ctx
            ));
        }
        
        // Tohumlar
        for (let i = 0; i < 40; i++) {
            this.elements.push(new Seed(
                Math.random() * this.canvas.width,
                Math.random() * this.canvas.height,
                this.ctx
            ));
        }
        
        // Bitki hücreleri
        for (let i = 0; i < 20; i++) {
            this.elements.push(new Cell(
                Math.random() * this.canvas.width,
                Math.random() * this.canvas.height,
                this.ctx,
                this
            ));
        }
        
        // Işık parıltıları
        for (let i = 0; i < 30; i++) {
            this.elements.push(new LightSparkle(
                Math.random() * this.canvas.width,
                Math.random() * this.canvas.height,
                this.ctx
            ));
        }
        
        // Grid Lines
        this.elements.push(new GridLines(this.ctx, this));
    }
    
    animate() {
        // Arka planı temizle
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        
        // Degrade arka plan 
        const gradient = this.ctx.createLinearGradient(0, 0, 0, this.canvas.height);
        gradient.addColorStop(0, 'rgba(20, 45, 40, 0.97)');
        gradient.addColorStop(0.5, 'rgba(12, 35, 32, 0.97)');
        gradient.addColorStop(1, 'rgba(8, 25, 22, 0.97)');
        
        this.ctx.fillStyle = gradient;
        this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
        
        // Background glow effect
        const radialGradient = this.ctx.createRadialGradient(
            this.canvas.width / 2, this.canvas.height / 2, 0,
            this.canvas.width / 2, this.canvas.height / 2, Math.max(this.canvas.width, this.canvas.height) / 1.5
        );
        radialGradient.addColorStop(0, 'rgba(40, 130, 100, 0.05)');
        radialGradient.addColorStop(1, 'rgba(0, 0, 0, 0)');
        
        this.ctx.fillStyle = radialGradient;
        this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
        
        // Tüm elementleri çiz ve animasyonu güncelle
        this.elements.forEach(element => {
            element.update();
            element.draw();
        });
        
        // Mouse interaction glow
        if (this.isMouseMoving && this.mouseSpeed > 0.5) {
            const size = 100 + this.mouseSpeed * 20;
            const glow = this.ctx.createRadialGradient(
                this.mouseX, this.mouseY, 0,
                this.mouseX, this.mouseY, size
            );
            glow.addColorStop(0, `rgba(60, 220, 150, ${Math.min(0.2, this.mouseSpeed * 0.04)})`);
            glow.addColorStop(1, 'rgba(60, 220, 150, 0)');
            
            this.ctx.fillStyle = glow;
            this.ctx.beginPath();
            this.ctx.arc(this.mouseX, this.mouseY, size, 0, Math.PI * 2);
            this.ctx.fill();
        }
        
        // Bir sonraki frame'i çiz
        requestAnimationFrame(() => this.animate());
    }
}

// Grid Lines - Işıyan ızgara çizgileri
class GridLines {
    constructor(ctx, parent) {
        this.ctx = ctx;
        this.parent = parent;
        this.spacing = 100;
        this.time = 0;
        this.speed = 0.5;
        
        // Grid lines
        this.horizontalLines = [];
        this.verticalLines = [];
        
        // Izgara çizgilerini oluştur
        this.createLines();
    }
    
    createLines() {
        const width = this.ctx.canvas.width;
        const height = this.ctx.canvas.height;
        
        // Horizontal lines
        for (let y = 0; y < height; y += this.spacing) {
            this.horizontalLines.push({
                y: y,
                opacity: 0.1 + Math.random() * 0.2,
                speed: 0.2 + Math.random() * 0.3,
                pulsePhase: Math.random() * Math.PI * 2
            });
        }
        
        // Vertical lines
        for (let x = 0; x < width; x += this.spacing) {
            this.verticalLines.push({
                x: x,
                opacity: 0.1 + Math.random() * 0.2,
                speed: 0.2 + Math.random() * 0.3,
                pulsePhase: Math.random() * Math.PI * 2
            });
        }
    }
    
    update() {
        this.time += 0.01;
        
        // Mouse interaction factor
        const interactionFactor = this.parent.mouseSpeed * 0.1;
        
        // Update horizontal lines
        this.horizontalLines.forEach(line => {
            line.opacity = 0.05 + 0.05 * Math.sin(this.time * line.speed + line.pulsePhase) + interactionFactor;
        });
        
        // Update vertical lines
        this.verticalLines.forEach(line => {
            line.opacity = 0.05 + 0.05 * Math.sin(this.time * line.speed + line.pulsePhase) + interactionFactor;
        });
    }
    
    draw() {
        const width = this.ctx.canvas.width;
        const height = this.ctx.canvas.height;
        
        // Draw horizontal lines
        this.horizontalLines.forEach(line => {
            this.ctx.beginPath();
            this.ctx.moveTo(0, line.y);
            this.ctx.lineTo(width, line.y);
            this.ctx.strokeStyle = `rgba(100, 255, 180, ${line.opacity})`;
            this.ctx.lineWidth = 0.5;
            this.ctx.stroke();
        });
        
        // Draw vertical lines
        this.verticalLines.forEach(line => {
            this.ctx.beginPath();
            this.ctx.moveTo(line.x, 0);
            this.ctx.lineTo(line.x, height);
            this.ctx.strokeStyle = `rgba(100, 255, 180, ${line.opacity})`;
            this.ctx.lineWidth = 0.5;
            this.ctx.stroke();
        });
    }
}

// DNA sarmalı sınıfı
class DNAHelix {
    constructor(x, y, ctx) {
        this.x = x;
        this.y = y;
        this.ctx = ctx;
        this.speed = 0.2 + Math.random() * 0.3;
        this.amplitude = 30 + Math.random() * 30;
        this.frequency = 0.05 + Math.random() * 0.05;
        this.length = 150 + Math.random() * 150;
        this.rotation = Math.random() * Math.PI * 2;
        this.time = Math.random() * 100;
        this.alpha = 0.5 + Math.random() * 0.3;
        
        // DNA renkleri
        this.colors = [
            'rgba(0, 180, 130, 0.7)',
            'rgba(60, 220, 150, 0.7)',
            'rgba(100, 200, 170, 0.7)'
        ];
    }
    
    update() {
        this.time += this.speed * 0.03;
        this.y -= this.speed;
        
        // Ekranın üstüne çıkarsa altan yeniden başla
        if (this.y < -this.length) {
            this.y = window.innerHeight + 50;
            this.x = Math.random() * window.innerWidth;
        }
    }
    
    draw() {
        this.ctx.save();
        
        this.ctx.translate(this.x, this.y);
        this.ctx.rotate(this.rotation);
        
        // DNA glow effect
        this.ctx.shadowBlur = 5;
        this.ctx.shadowColor = 'rgba(0, 255, 150, 0.3)';
        
        for (let i = 0; i < this.length; i += 8) {
            const wave1X = Math.sin(i * this.frequency + this.time) * this.amplitude;
            const wave2X = Math.sin(i * this.frequency + this.time + Math.PI) * this.amplitude;
            
            // Sol helix
            this.ctx.beginPath();
            this.ctx.arc(wave1X, i, 2, 0, Math.PI * 2);
            this.ctx.fillStyle = this.colors[i % 3];
            this.ctx.fill();
            
            // Sağ helix
            this.ctx.beginPath();
            this.ctx.arc(wave2X, i, 2, 0, Math.PI * 2);
            this.ctx.fillStyle = this.colors[(i + 1) % 3];
            this.ctx.fill();
            
            // Bağlantılar
            if (i % 20 === 0) {
                this.ctx.beginPath();
                this.ctx.moveTo(wave1X, i);
                this.ctx.lineTo(wave2X, i);
                this.ctx.strokeStyle = 'rgba(100, 200, 170, 0.4)';
                this.ctx.lineWidth = 1;
                this.ctx.stroke();
            }
        }
        
        this.ctx.restore();
    }
}

// Yaprak deseni sınıfı
class LeafPattern {
    constructor(x, y, ctx) {
        this.x = x;
        this.y = y;
        this.ctx = ctx;
        this.rotation = 0;
        this.rotationSpeed = 0.001 + Math.random() * 0.002;
        this.scale = 0.2 + Math.random() * 0.6;
        this.growSpeed = 0.0005 + Math.random() * 0.001;
        this.alpha = 0;
        this.fadeSpeed = 0.003 + Math.random() * 0.007;
        
        // Renk varyasyonu
        this.hue = 120 + Math.floor(Math.random() * 60); // Yeşil tonları
        this.color = `hsla(${this.hue}, 80%, 50%, 0.6)`;
    }
    
    update() {
        this.rotation += this.rotationSpeed;
        this.scale += this.growSpeed;
        this.alpha += this.fadeSpeed;
        
        // Belirli bir büyüklüğe ve opaklığa ulaşınca sıfırla
        if (this.scale > 1.2 || this.alpha > 1) {
            this.scale = 0.2 + Math.random() * 0.2;
            this.x = Math.random() * window.innerWidth;
            this.y = Math.random() * window.innerHeight;
            this.alpha = 0;
        }
    }
    
    draw() {
        this.ctx.save();
        
        this.ctx.translate(this.x, this.y);
        this.ctx.rotate(this.rotation);
        this.ctx.scale(this.scale, this.scale);
        
        // Yaprak glow effect
        this.ctx.shadowBlur = 10;
        this.ctx.shadowColor = `hsla(${this.hue}, 80%, 50%, ${this.alpha * 0.3})`;
        
        // Yaprak çizimi
        this.ctx.beginPath();
        
        this.ctx.moveTo(0, -50);
        this.ctx.bezierCurveTo(25, -40, 40, -30, 50, 0);
        this.ctx.bezierCurveTo(40, 30, 25, 40, 0, 50);
        this.ctx.bezierCurveTo(-25, 40, -40, 30, -50, 0);
        this.ctx.bezierCurveTo(-40, -30, -25, -40, 0, -50);
        
        this.ctx.fillStyle = `hsla(${this.hue}, 80%, 50%, ${this.alpha * 0.4})`;
        this.ctx.fill();
        
        // Yaprak damarları
        this.ctx.beginPath();
        this.ctx.moveTo(0, -50);
        this.ctx.lineTo(0, 50);
        this.ctx.strokeStyle = `hsla(${this.hue}, 80%, 40%, ${this.alpha * 0.5})`;
        this.ctx.lineWidth = 1;
        this.ctx.stroke();
        
        // Yan damarlar
        for (let i = -40; i <= 40; i += 10) {
            this.ctx.beginPath();
            this.ctx.moveTo(0, i);
            this.ctx.lineTo(i > 0 ? 25 : -25, i);
            this.ctx.strokeStyle = `hsla(${this.hue}, 80%, 40%, ${this.alpha * 0.3})`;
            this.ctx.lineWidth = 0.5;
            this.ctx.stroke();
        }
        
        this.ctx.restore();
    }
}

// Tohum sınıfı
class Seed {
    constructor(x, y, ctx) {
        this.x = x;
        this.y = y;
        this.ctx = ctx;
        this.rotation = Math.random() * Math.PI * 2;
        this.rotationSpeed = 0.01 + Math.random() * 0.03;
        this.xVelocity = Math.random() * 0.5 - 0.25;
        this.yVelocity = -0.4 - Math.random() * 0.6;
        this.size = 2 + Math.random() * 5;
        this.alpha = 0.6 + Math.random() * 0.4;
        
        // Trail effect
        this.trail = [];
        this.trailLength = 5 + Math.floor(Math.random() * 10);
    }
    
    update() {
        // Update position
        this.x += this.xVelocity;
        this.y += this.yVelocity;
        this.rotation += this.rotationSpeed;
        
        // Add current position to trail
        this.trail.push({x: this.x, y: this.y, alpha: this.alpha});
        
        // Limit trail length
        if (this.trail.length > this.trailLength) {
            this.trail.shift();
        }
        
        // Ekran dışına çıkarsa yeniden konumlandır
        if (this.y < -20 || this.x < -20 || this.x > window.innerWidth + 20) {
            this.y = window.innerHeight + 10;
            this.x = Math.random() * window.innerWidth;
            this.xVelocity = Math.random() * 0.5 - 0.25;
            this.yVelocity = -0.4 - Math.random() * 0.6;
            this.trail = []; // Reset trail
        }
    }
    
    draw() {
        // Draw trail
        for (let i = 0; i < this.trail.length; i++) {
            const t = this.trail[i];
            const fadeRatio = i / this.trail.length;
            const alpha = t.alpha * fadeRatio * 0.5;
            
            this.ctx.save();
            this.ctx.translate(t.x, t.y);
            this.ctx.rotate(this.rotation);
            
            // Tohum glow effect
            this.ctx.shadowBlur = 5;
            this.ctx.shadowColor = `rgba(220, 255, 220, ${alpha})`;
            
            // Tohum trail
            this.ctx.beginPath();
            this.ctx.ellipse(0, 0, this.size * fadeRatio, this.size * 2 * fadeRatio, 0, 0, Math.PI * 2);
            this.ctx.fillStyle = `rgba(220, 255, 220, ${alpha})`;
            this.ctx.fill();
            
            this.ctx.restore();
        }
        
        // Draw seed
        this.ctx.save();
        
        this.ctx.translate(this.x, this.y);
        this.ctx.rotate(this.rotation);
        
        // Tohum glow effect
        this.ctx.shadowBlur = 8;
        this.ctx.shadowColor = `rgba(220, 255, 220, ${this.alpha * 0.5})`;
        
        // Tohum çizimi
        this.ctx.beginPath();
        this.ctx.ellipse(0, 0, this.size, this.size * 2, 0, 0, Math.PI * 2);
        this.ctx.fillStyle = `rgba(230, 255, 230, ${this.alpha})`;
        this.ctx.fill();
        
        // Tohum detayları
        this.ctx.beginPath();
        this.ctx.ellipse(0, 0, this.size * 0.6, this.size * 1.6, 0, 0, Math.PI * 2);
        this.ctx.fillStyle = `rgba(180, 255, 200, ${this.alpha * 0.8})`;
        this.ctx.fill();
        
        this.ctx.restore();
    }
}

// Bitki hücresi sınıfı
class Cell {
    constructor(x, y, ctx, parent) {
        this.x = x;
        this.y = y;
        this.ctx = ctx;
        this.parent = parent;
        this.rotation = Math.random() * Math.PI * 2;
        this.rotationSpeed = 0.002 + Math.random() * 0.005;
        this.scale = 0.2 + Math.random() * 0.8;
        this.alpha = 0.2 + Math.random() * 0.4;
        this.pulseSpeed = 0.01 + Math.random() * 0.02;
        this.time = Math.random() * 100;
        this.hue = 140 + Math.floor(Math.random() * 40);
        
        // Cell connections
        this.connections = [];
        this.maxConnections = 3;
    }
    
    update() {
        this.time += this.pulseSpeed;
        this.rotation += this.rotationSpeed;
        
        // Hücre yavaşça hareket etsin
        this.x += Math.sin(this.time * 0.1) * 0.2;
        this.y += Math.cos(this.time * 0.1) * 0.2;
        
        // Mouse interaction
        if (this.parent.isMouseMoving) {
            const dx = this.parent.mouseX - this.x;
            const dy = this.parent.mouseY - this.y;
            const distance = Math.sqrt(dx * dx + dy * dy);
            
            if (distance < 200) {
                const force = (1 - distance / 200) * this.parent.mouseSpeed * 0.2;
                this.x += dx * force / distance;
                this.y += dy * force / distance;
            }
        }
        
        // Ekran dışına çıkarsa yeniden konumlandır
        if (this.x < -50 || this.x > window.innerWidth + 50 || 
            this.y < -50 || this.y > window.innerHeight + 50) {
            this.x = Math.random() * window.innerWidth;
            this.y = Math.random() * window.innerHeight;
        }
    }
    
    draw() {
        this.ctx.save();
        
        this.ctx.translate(this.x, this.y);
        this.ctx.rotate(this.rotation);
        
        const pulse = 1 + Math.sin(this.time) * 0.1;
        this.ctx.scale(this.scale * pulse, this.scale * pulse);
        
        // Cell glow effect
        this.ctx.shadowBlur = 10;
        this.ctx.shadowColor = `hsla(${this.hue}, 90%, 50%, ${this.alpha * 0.5})`;
        
        // Hücre duvarı
        this.ctx.beginPath();
        this.ctx.arc(0, 0, 30, 0, Math.PI * 2);
        this.ctx.fillStyle = `hsla(${this.hue}, 90%, 50%, ${this.alpha * 0.4})`;
        this.ctx.fill();
        
        this.ctx.beginPath();
        this.ctx.arc(0, 0, 28, 0, Math.PI * 2);
        this.ctx.fillStyle = `hsla(${this.hue}, 90%, 60%, ${this.alpha * 0.3})`;
        this.ctx.fill();
        
        // Hücre içi
        this.ctx.beginPath();
        this.ctx.arc(0, 0, 25, 0, Math.PI * 2);
        this.ctx.fillStyle = `hsla(${this.hue}, 80%, 40%, ${this.alpha * 0.2})`;
        this.ctx.fill();
        
        // Hücre çekirdeği
        this.ctx.beginPath();
        this.ctx.arc(0, 0, 10, 0, Math.PI * 2);
        this.ctx.fillStyle = `hsla(${this.hue - 10}, 90%, 30%, ${this.alpha * 0.6})`;
        this.ctx.fill();
        
        // Mitokondri ve diğer organeller
        for (let i = 0; i < 5; i++) {
            const angle = i * Math.PI * 2 / 5 + this.time * 0.2;
            const distance = 15;
            const x = Math.cos(angle) * distance;
            const y = Math.sin(angle) * distance;
            
            this.ctx.beginPath();
            this.ctx.ellipse(x, y, 3, 5, angle, 0, Math.PI * 2);
            this.ctx.fillStyle = `hsla(${this.hue + 20}, 90%, 60%, ${this.alpha * 0.5})`;
            this.ctx.fill();
        }
        
        this.ctx.restore();
    }
}

// Işık parıltısı sınıfı
class LightSparkle {
    constructor(x, y, ctx) {
        this.x = x;
        this.y = y;
        this.ctx = ctx;
        this.size = 20 + Math.random() * 60;
        this.alpha = 0;
        this.fadeSpeed = 0.005 + Math.random() * 0.01;
        this.maxAlpha = 0.1 + Math.random() * 0.2;
        this.hue = 160 + Math.floor(Math.random() * 60);
    }
    
    update() {
        this.alpha += this.fadeSpeed;
        
        // Belirli bir opaklığa ulaşınca sıfırla
        if (this.alpha > this.maxAlpha * 2) {
            this.alpha = 0;
            this.x = Math.random() * window.innerWidth;
            this.y = Math.random() * window.innerHeight;
            this.size = 20 + Math.random() * 60;
        }
    }
    
    draw() {
        this.ctx.save();
        
        const currentAlpha = this.alpha > this.maxAlpha ? 
                            this.maxAlpha * 2 - this.alpha : this.alpha;
        
        // Işık parlama efekti
        const gradient = this.ctx.createRadialGradient(
            this.x, this.y, 0,
            this.x, this.y, this.size
        );
        
        gradient.addColorStop(0, `hsla(${this.hue}, 100%, 80%, ${currentAlpha * 1.5})`);
        gradient.addColorStop(0.5, `hsla(${this.hue}, 100%, 60%, ${currentAlpha * 0.5})`);
        gradient.addColorStop(1, `hsla(${this.hue}, 100%, 50%, 0)`);
        
        this.ctx.fillStyle = gradient;
        this.ctx.beginPath();
        this.ctx.arc(this.x, this.y, this.size, 0, Math.PI * 2);
        this.ctx.fill();
        
        this.ctx.restore();
    }
}

// Sayfa yüklendiğinde arka planı başlat
window.addEventListener('load', () => {
    const background = new BotanicalTechBackground();
}); 