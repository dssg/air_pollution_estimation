import smtplib
import ssl

from traffic_analysis.d00_utils.load_confs import load_credentials


def send_email_warning(msg: str = 'The script responsible for downloading the traffic camera data has been stopped. Please check EC2 instance.',
                       subj: str = 'ERROR - Data Download Failed',
                       port=465,
                       smtp_server="smtp.gmail.com"):
    try:
        creds = load_credentials()
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
    except Exception as e:
        print(
            f"Failed to send email {message}. Please check that the email key is added to the credentials file. For instance, \nemail:\n\taddress:\n\tpassword:\n\trecipients:[]")
        print("Reason: ", e)
