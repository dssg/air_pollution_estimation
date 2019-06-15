import smtplib
import ssl
import configparser
import os
import ast

def send_email():
    setup_dir = os.path.join(os.getcwd(), '..', '..')
    config = configparser.ConfigParser()
    config.read(os.path.join(setup_dir, 'conf', 'local', 'credentials.yml'))
    print("config >>>>>>>>>>>>> ",config.values())
    sender_email = config.get('EMAIL', 'address')
    password = config.get('EMAIL', 'password')
    recipients = ast.literal_eval(config.get('EMAIL', 'recipients'))

    port = 465  # For SSL
    smtp_server = "smtp.gmail.com"
    message = """\
    Subject: ERROR - Traffic Camera Download Failed

    The script responsible for downloading the traffic camera data has been stopped."""

    context = ssl.create_default_context()
    with smtplib.SMTP_SSL(smtp_server, port, context=context) as server:
        server.login(sender_email, password)
        server.sendmail(sender_email, recipients, message)
