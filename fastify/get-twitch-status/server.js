module.exports = function (fastify, opts, done) {
    const axios = opts.axios;
    const twitchAPIKey = opts.twitchAPIKey;
    const twitchSecretKey = opts.twitchSecretKey;
    
    fastify.get('/get-twitch-status', async (request, reply) => {
        let twitchStatus = false;

        const client_id = twitchAPIKey;
        const client_secret = twitchSecretKey;
        const streamer_name = 'stocknear';

        try {
            // Obtain an access token from Twitch
            
            const tokenResponse = await axios.post('https://id.twitch.tv/oauth2/token', null, {
                params: {
                    client_id,
                    client_secret,
                    grant_type: 'client_credentials',
                },
            });

            const { access_token } = tokenResponse.data;

            // Check if the stream is online
            const streamResponse = await axios.get(
                `https://api.twitch.tv/helix/streams?user_login=${streamer_name}`,
                {
                    headers: {
                        'Client-ID': client_id,
                        Authorization: `Bearer ${access_token}`,
                    },
                }
            );

            const streamData = streamResponse.data;
            twitchStatus = streamData.data.length === 1;
            

        } catch (e) {
            console.error(e);
        }

        reply.send({ items: twitchStatus });
    });

    done();
};
