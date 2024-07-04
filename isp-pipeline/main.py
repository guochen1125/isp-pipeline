from src import pipeline

if __name__ == '__main__':
    isp = pipeline('config.yml')
    y = isp.run()
