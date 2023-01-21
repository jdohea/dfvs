from dfvs_basic_branching import DFVS
from dfvs_mcts import MCTSSolver
import time
import os
import sys
from discord_webhook import DiscordWebhook
import smtplib
from os.path import basename
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import csv

files = ['test2']
solvers = [DFVS(), MCTSSolver()]
results = []


def experiment():
    inputdir = 'graphs/exact_public'
    send_discord_message('Starting on directory: ' + inputdir)
    blahblahblah = os.listdir(inputdir)
    blahblahblah.sort()
    results.append(['file_name, sovler_name,time_taken', 'm_dfvs_length'])
    for file in blahblahblah:
        for solver in solvers:
            before = time.time()
            m_dfvs_length = len(solver.exact_solver(inputdir + '/' + file))
            after = time.time()
            time_taken = after - before
            x = [str(file), solver.name, time_taken, m_dfvs_length]
            results.append(x)
            send_discord_message(str(x))

    with open("results.csv", "w+") as my_csv:
        csvWriter = csv.writer(my_csv, delimiter=',')
        csvWriter.writerows(results)
    send_results()


def send_discord_message(message):
    url = os.getenv('discord_webhook_url')
    webhook = DiscordWebhook(url=url, rate_limit_retry=True, content=message)
    return webhook.execute()


def send_results():
    sender = 'jdleisureinformation@gmail.com'
    receiver = 'jdohea@gmail.com'
    bodySend = ""
    password ='viisblvrswqnxvah'

    msg = MIMEMultipart()
    msg['Subject'] = 'dfvs results'
    msg['From'] = sender
    msg['To'] = receiver
    files = ['results.csv']
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


# if __name__ == "__main__":
    sys.setrecursionlimit(10**7)
    experiment()
    send_discord_message('finished ')


