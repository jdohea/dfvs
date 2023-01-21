
import smtplib
from os.path import basename
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

def send_results(files = []):
    if len(files) ==0:
        files = ['results.csv']
    sender = 'jdleisureinformation@gmail.com'
    receiver = 'jdohea@gmail.com'
    bodySend = ""
    password ='viisblvrswqnxvah'

    msg = MIMEMultipart()
    msg['Subject'] = 'mcts_depth'
    msg['From'] = sender
    msg['To'] = receiver
    msg.attach(MIMEText(bodySend))
    for f in files or []:
            with open(f, "rb") as fil:
                part = MIMEApplication(
                    fil.read(),
                    Name=basename(f)
                )
            # After the file is closed
            part['Content-Disposition'] = 'attachment; filename="%s"' % basename(f)
            msg.attach(part)

    s = smtplib.SMTP_SSL(host = 'smtp.gmail.com', port = 465)
    s.login(user = sender, password = password)
    s.sendmail(sender, receiver, msg.as_string())
    s.quit()


if __name__ == "__main__":
    send_results()