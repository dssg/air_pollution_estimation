import smtplib
import ssl

from traffic_analysis.d00_utils.load_confs import load_credentials

creds = load_credentials()


def send_email_warning(msg: str = 'The script responsible for downloading the traffic camera data has been stopped. Please check EC2 instance.',
                       subj: str = 'ERROR - Data Download Failed',
                       port=465,
                       smtp_server="smtp.gmail.com"):
    sender_email = creds['email']['address']
    password = creds['email']['password']
    recipients = creds['email']['recipients']
    subject = subj
    text = msg
    message = 'Subject: {}\n\n{}'.format(subject, text)

    context = ssl.create_default_context()
    with smtplib.SMTP_SSL(smtp_server, port, context=context) as server:
        server.login(sender_email, password)
        server.sendmail(sender_email, recipients, message)
    return
