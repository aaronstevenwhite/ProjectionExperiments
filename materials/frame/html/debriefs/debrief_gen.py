from string import hexdigits
from random import sample

num_of_files= 30

for i in range(num_of_files):
	f = open('frame_debrief' + str(i) + '.html', 'w')

	key = ''.join([str(sample(hexdigits, 1)[0]) for i in range(15)])

	html = '''<div style="width: 40em;">
		<p>
			You are now finished with the experiment. Your payment code can be found below. You will copy this code into the <i>Code</i> field of the survey page in Amazon Mechanical Turk. Once you have done so, submit the survey, check the box below, and click continue. <i>Don't forget to do the last step, otherwise we cannot credit your account.</i> Once we have received both the code from Amazon and your data, you will receive your credit.  
		</p>

		<center>
			<b>{}</b>
		</center>

		<p>
			<input name="consent" type="radio" value="consent_checked" class="obligatory" id="consent_info" /><label for="consent_info">I have entered the code into Mechanical Turk. Submit my data!</label>
		</p>
	</div>'''.format(key)

	f.write(html)
	
	f.close()
