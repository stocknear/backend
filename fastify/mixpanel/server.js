
// Declare a route
module.exports = function (fastify, opts, done) {
    
    const mixpanel = opts.mixpanel;
    const UAParser = opts.UAParser;

    fastify.post('/mixpanel', async (request, reply) => {

    const data = request.body;

    const { browser, cpu, device, os } = UAParser(data.userAgent)

    let options = { 
        path: data.path,
        browser: browser.name,
        browser_version: browser.version,
        device: device.vendor,
        cpu: cpu.architecture,
        os: os.name,
    }


    if (data.type === 'trackPageError')
    {
      options.status = data.status;
      options.message = data.message;
      mixpanel.track('Error status', options); 
      console.log('Send error page data to mixpanel')
    }
  
    else if (data.type === 'trackPageVisit')
    {
      mixpanel.track('Page Visit', options); 
  
    }
  
    else if (data.type === 'trackPageDuration')
    {
      options.time_spent = data.time;
  
      mixpanel.track('Page Duration', options);
  
    }

    else if (data.type === 'trackAsset')
    {
      const options = {
        symbol: data.symbol,
        assetType: data.assetType,
      }
  
      mixpanel.track('asset', options);
  
    }

    else if (data.type === 'trackButton')
    {
      const options = {
        name: data.name,
      }
  
      mixpanel.track('buttonClick', options);
  
    }

    reply.send({ message: 'success' })
    })

    done();
};

