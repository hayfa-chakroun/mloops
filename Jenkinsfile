pipeline {
    agent any

    stages {
        stage('Cloner le projet') {
            steps {
                git 'https://github.com/hayfa211/mlops_pipeline.git'
            }
        }
        stage('Installer les dépendances') {
            steps {
                sh 'pip install -r requirements.txt'
            }
        }
        stage('Lancer l’API FastAPI') {
            steps {
                sh 'uvicorn app:app --host 0.0.0.0 --port 8001 --reload &'
            }
        }
        stage('Tester l’API') {
            steps {
                sh 'curl -X GET http://127.0.0.1:8001'
            }
        }
        stage('Construire et exécuter Docker') {
            steps {
                sh 'docker build -t mlops-app .'
                sh 'docker run -p 8001:8001 mlops-app'
            }
        }
    }
}

