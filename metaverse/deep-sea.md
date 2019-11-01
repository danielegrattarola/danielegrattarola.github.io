---
layout: metaverse
title: deep sea
image: /images/metaverse/1.jpg
extra_css: 
    - css/base.css
---

<div class="image">
    <canvas id="can"></canvas>
</div>

<div class="text">
    Generated with <3 by
    <pre>
red: function(i, j) {
    var a=0,b=0,d=0,n=0;
    for(;a*a+(d=b*b)<4&&n++<1024;b=2*a*b+j/5e4+.06,a=a*a-d+i/5e4+.34);
    return n/4;
}

green: function(i, j) {
    return 4 * this.red(i, j);
}

blue: function(i, j) {
    return 6 * this.red(i, j)
}
    </pre>
</div>

<script type="text/javascript">

	var def = {
		size: 700,
		red: function(i, j) {
			var a=0,b=0,d=0,n=0;
			for(;a*a+(d=b*b)<4&&n++<1024;b=2*a*b+j/5e4+.06,a=a*a-d+i/5e4+.34);
			return n/4;
		},

		green: function(i, j) {
			return 4 * this.red(i,j);
		},

		blue: function(i, j) {
			return 6 * this.red(i,j)
		}
	}

	function draw(f) {
		var can = document.getElementById('can');
		can.width = can.height = f.size;
		var ctx = can.getContext('2d');
		ctx.fillRect(0, 0, f.size, f.size);
		var imgData = ctx.getImageData(0, 0, f.size, f.size);
		var data = imgData.data;
		for (var i = 0; i < data.length; i += 4) {
			var i2 = (i / 4) % f.size
			var j2 = Math.floor(i / 4 / f.size);
			data[i] = f.red(i2, j2) % 256;
			data[i + 1] = f.green(i2, j2) % 256;
			data[i + 2] = f.blue(i2, j2) % 256;
		}
		ctx.putImageData(imgData, 0, 0);
	}

	draw(def);

</script>
