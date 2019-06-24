import smtplib
import ssl
import configparser
import os
import ast


def send_email_warning(msg: str = 'The script responsible for downloading the traffic camera data has been stopped. Please check EC2 instance.',
                       subj: str = 'ERROR - Data Download Failed'):
    setup_dir = os.path.join(os.path.dirname(
        os.path.abspath(__file__)), '..', '..')
    config = configparser.ConfigParser()
    config.read(os.path.join(setup_dir, 'conf', 'local', 'credentials.yml'))

    sender_email = config.get('EMAIL', 'address')
    password = config.get('EMAIL', 'password')
    recipients = ast.literal_eval(config.get('EMAIL', 'recipients'))

    port = 465  # For SSL
    smtp_server = "smtp.gmail.com"
    subject = subj
    text = msg
    message = 'Subject: {}\n\n{}'.format(subject, text)

    context = ssl.create_default_context()
    with smtplib.SMTP_SSL(smtp_server, port, context=context) as server:
        server.login(sender_email, password)
        server.sendmail(sender_email, recipients, message)
    return
