pipeline {
    agent any

    environment {
        EC2_USER   = 'ec2-user'
        EC2_IP     = '3.71.6.247'
        DEPLOY_PATH = '/home/ec2-user/football_bet'
        KEY_PATH    = '/var/lib/jenkins/.ssh/nurludursun.pem'
    }

    stages {
        stage('Checkout') {
            steps {
                checkout scm
            }
        }
        stage('Build') {
            steps {
                sh "echo 'Build aşaması... (Örnek: mvn clean install)'"
            }
        }
        stage('Deploy to EC2') {
            steps {
                sh """
                    echo 'Deploy aşaması...'
                    scp -o StrictHostKeyChecking=no -i ${KEY_PATH} -r * ${EC2_USER}@${EC2_IP}:${DEPLOY_PATH}
                """
            }
        }
    }

    post {
        success {
            echo "Pipeline başarıyla tamamlandı!"
        }
        failure {
            echo "Pipeline sırasında hata oluştu!!"
        }
    }
}
