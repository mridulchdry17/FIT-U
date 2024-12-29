workers = 1
worker_class = 'gevent'  # Use gevent for better async performance
timeout = 300
bind = "0.0.0.0:10000"
max_requests = 1000
max_requests_jitter = 50 