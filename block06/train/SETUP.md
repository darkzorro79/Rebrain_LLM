# 🔧 Подготовка системы Ubuntu 24.04

Инструкции для администратора по подготовке виртуальной машины с NVIDIA H100.

## 1. Проверка системы

```bash
# Проверка версии Ubuntu
lsb_release -a

# Проверка GPU
lspci | grep -i nvidia

# Проверка драйверов NVIDIA
nvidia-smi
```

Ожидаемый результат `nvidia-smi`:
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 535.xx.xx    Driver Version: 535.xx.xx    CUDA Version: 12.x   |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  NVIDIA H100         Off  | 00000000:00:1E.0 Off |                    0 |
| N/A   35C    P0    70W / 350W |      0MiB / 98304MiB |      0%      Default |
...
```

## 2. Установка Python 3.12 (если отсутствует)

```bash
# Обновление системы
sudo apt update && sudo apt upgrade -y

# Установка Python 3.12
sudo apt install python3.12 python3.12-venv python3.12-dev -y

# Проверка версии
python3.12 --version
```

## 3. Установка pip и утилит

```bash
# Установка pip
sudo apt install python3-pip -y

# Установка системных зависимостей
sudo apt install build-essential git wget curl -y
```

## 4. Проверка CUDA

```bash
# Проверка версии CUDA
nvcc --version

# Если CUDA не установлена, установите CUDA Toolkit 12.x:
# https://developer.nvidia.com/cuda-downloads
```

## 5. Создание директорий проекта

```bash
# Создание директории /opt/train с правильными правами
sudo mkdir -p /opt/train
sudo chown -R $USER:$USER /opt/train
sudo chmod -R 755 /opt/train
```

## 6. Установка PyTorch с поддержкой CUDA

```bash
# Активируем venv
source /opt/train/venv/bin/activate

# Установка PyTorch с CUDA 12.1 (или вашей версии)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Проверка установки:

```python
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'Device count: {torch.cuda.device_count()}')"
```

Ожидаемый вывод:
```
PyTorch: 2.1.x+cu121
CUDA available: True
CUDA version: 12.1
Device count: 1
```

## 7. Настройка SSH (если нужен удаленный доступ)

```bash
# Установка OpenSSH Server (если не установлен)
sudo apt install openssh-server -y

# Запуск службы SSH
sudo systemctl start ssh
sudo systemctl enable ssh

# Проверка статуса
sudo systemctl status ssh
```

Создание SSH ключа для входа:

```bash
# На локальной машине
ssh-keygen -t ed25519 -C "your_email@example.com"

# Копирование ключа на сервер
ssh-copy-id user@server_ip
```

## 8. Настройка файрвола (опционально)

```bash
# Установка UFW
sudo apt install ufw -y

# Разрешение SSH
sudo ufw allow ssh

# Разрешение порта для vLLM API (8000)
sudo ufw allow 8000/tcp

# Включение файрвола
sudo ufw enable

# Проверка статуса
sudo ufw status
```

## 9. Мониторинг GPU в реальном времени

```bash
# Установка gpustat (опционально, но удобно)
pip install gpustat

# Запуск мониторинга
watch -n 1 gpustat
```

Или стандартная команда:

```bash
watch -n 1 nvidia-smi
```

## 10. Проверка готовности системы

Чек-лист перед началом обучения:

```bash
# 1. Python 3.12
python3.12 --version

# 2. NVIDIA Driver
nvidia-smi

# 3. CUDA
nvcc --version

# 4. PyTorch с CUDA
python3 -c "import torch; print(torch.cuda.is_available())"

# 5. Директории
ls -la /opt/train

# 6. Права доступа
touch /opt/train/test.txt && rm /opt/train/test.txt && echo "Права OK"
```

## 11. Оптимизация производительности (опционально)

### Отключение энергосбережения GPU

```bash
# Установка persistence mode (сохраняется до перезагрузки)
sudo nvidia-smi -pm 1

# Максимальная производительность
sudo nvidia-smi -pl 350  # Для H100 (max power)
```

### Настройка системных лимитов

Добавьте в `/etc/security/limits.conf`:

```
* soft nofile 65536
* hard nofile 65536
* soft nproc 65536
* hard nproc 65536
```

### Оптимизация памяти

Добавьте в `/etc/sysctl.conf`:

```
vm.swappiness=10
vm.overcommit_memory=1
```

Применить:

```bash
sudo sysctl -p
```

## 12. Настройка автозапуска (опционально)

Создайте systemd service для автозапуска vLLM:

```bash
sudo nano /etc/systemd/system/pirate-llm.service
```

Содержимое:

```ini
[Unit]
Description=Pirate LLM vLLM Server
After=network.target

[Service]
Type=simple
User=your_username
WorkingDirectory=/home/darkzorro/LLM_rebrain/block06/train
Environment="PATH=/opt/train/venv/bin"
ExecStart=/home/darkzorro/LLM_rebrain/block06/train/run_server.sh
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

Активация:

```bash
sudo systemctl daemon-reload
sudo systemctl enable pirate-llm.service
sudo systemctl start pirate-llm.service
```

## 13. Backup и восстановление

### Создание бэкапа весов LORA

```bash
# Создание архива
tar -czf /opt/train/backup_$(date +%Y%m%d).tar.gz /opt/train/qwen_final

# Копирование на удаленный сервер
scp /opt/train/backup_*.tar.gz user@backup-server:/backups/
```

### Восстановление

```bash
# Распаковка
tar -xzf /opt/train/backup_YYYYMMDD.tar.gz -C /
```

## 14. Логирование

Создайте директорию для логов:

```bash
sudo mkdir -p /var/log/pirate-llm
sudo chown $USER:$USER /var/log/pirate-llm
```

Перенаправление логов в `run_server.sh`:

```bash
vllm serve ... 2>&1 | tee /var/log/pirate-llm/server.log
```

## ✅ Система готова!

После выполнения всех шагов переходите к `QUICKSTART.md` для начала обучения модели.
